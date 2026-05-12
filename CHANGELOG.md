# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - Phase 26

### Added
- **CUDA FP8 E4M3/E5M2 batch prefill** (`oxibonsai-kernels`, `oxibonsai-model`): `try_cuda_prefill_fp8` replaces sequential single-token CUDA GEMV loop for FP8E4M3 and FP8E5M2 models. 8 new NVRTC kernels (gemm, gemm_residual, fused_gate_up_swiglu, gemv_pf × E4M3/E5M2) with cap-of-8 outer loop and correct FP8 block layout (weights at bytes 0-31, FP16 scale at bytes 32-33). New `forward_cuda_fp8.rs` module with `try_cuda_prefill_with_lm_head_fp8` / `try_cuda_prefill_verify_fp8`; handle namespaces E4M3 26M-31M, E5M2 28M-33M. `BonsaiModel::forward_prefill` and `forward_prefill_verify` now route FP8 models through batch GEMM (with sequential CUDA GEMV fallback on error). 4 host-only kernel-source-string tests added.

---

## [Unreleased] - Phase 25

### Added
- **K-quant CUDA batch prefill** (`oxibonsai-kernels`, `oxibonsai-model`): `try_cuda_prefill_k_quant` replaces sequential single-token CUDA GEMV loop for Q2K, Q3K, Q4K, Q5K, Q6K, and Q8K models. 18 new NVRTC kernels (3 per format: `gemm_q{fmt}`, `gemm_q{fmt}_residual`, `fused_gate_up_swiglu_gemm_q{fmt}`) with cap-of-8 outer loop preventing silent batch-size truncation. `KQuantFormat` enum for dispatch. `BonsaiModel::forward_prefill` and `forward_prefill_verify` now route K-quant models through `try_cuda_prefill_with_lm_head_k_quant` / `try_cuda_prefill_verify_k_quant` on CUDA hosts (with sequential fallback on error). Handle namespaces: Q2K norms `12M`, weights `13M`; Q3K `14M/15M`; Q4K `16M/17M`; Q5K `18M/19M`; Q6K `20M/21M`; Q8K `22M/23M`; final-norm `24M+fmt_offset`; LM-head `25M+fmt_offset`. 42 K-quant block accessor methods added to `TransformerBlock`. Runtime validation: `hidden_size % 256 == 0` required for all K-quant batch GEMM.

---

## [Unreleased] - Phase 24

### Added
- **CUDA Q4_0/Q8_0 batch prefill** (`oxibonsai-kernels`, `oxibonsai-model`): `try_cuda_prefill_q_std` replaces sequential single-token prefill with fused batch GEMM for Q4_0 and Q8_0 models. `BonsaiModel::forward_prefill` now routes Q4_0/Q8_0 through `try_cuda_prefill_with_lm_head_q_std` (with sequential fallback on error). `try_cuda_prefill_verify` routes Q4_0/Q8_0 through `try_cuda_prefill_verify_q_std`. Handle namespaces: norm `8M/10M`, weight `9M/11M` (Q4_0/Q8_0 respectively), final-norm `8.9M/10.9M`, LM-head `9.9M/11.9M`.

---

## [Unreleased] - Phase 23

### Added
- **K-quant CUDA GEMV kernels** (`oxibonsai-kernels`, `oxibonsai-model`): Six new NVRTC GEMV kernels for Q2K, Q3K, Q4K, Q5K, Q6K, and Q8K K-quant formats. `LinearQ2K/Q3K/Q4K/Q5K/Q6K/Q8K::forward` now dispatch to `cuda_gemv_q*k` NVRTC kernels when a CUDA device is available (Linux/Windows, `native-cuda` feature), falling back to CPU scalar on error. `BonsaiModel::forward` and `forward_prefill` detect `OutputWeight::Q2K/Q3K/Q4K/Q5K/Q6K/Q8K` + CUDA availability and route through the block dispatch loop, bypassing the Q1-only CUDA graph attempt. Compile-time `Block*` layout assertions added.

---

## [Unreleased] - Phase 22

### Added
- **Q4_0/Q8_0 CUDA full-forward dispatch** (`oxibonsai-model`, `oxibonsai-kernels`): `LinearQ4_0::forward` and `LinearQ8_0::forward` now dispatch to `cuda_gemv_q4_0`/`cuda_gemv_q8_0` NVRTC kernels when a CUDA device is available (Linux/Windows, `native-cuda` feature), falling back silently to CPU scalar on error. `BonsaiModel::forward` and `forward_prefill` detect `OutputWeight::Q4_0/Q8_0` + CUDA availability and route directly through the block dispatch loop, bypassing the Q1-only CUDA graph attempt. Includes `tracing::debug!` profile logging and compile-time `BlockQ4_0`/`BlockQ8_0` layout assertions.

---

## [Unreleased] - Phase 21

### Added
- **FP8 CUDA early-dispatch** (`oxibonsai-model`): `BonsaiModel::forward` and `forward_prefill` now detect `OutputWeight::FP8E4M3/E5M2` + CUDA availability and route directly through the block dispatch loop (which calls `cuda_gemv_fp8_e4m3/e5m2` via `KernelTier::Gpu`), bypassing the Q1-only CUDA graph attempt entirely. Includes `tracing::debug!` profile logging.
- **Block accessor methods** (`oxibonsai-model`): 14 FP8 E4M3/E5M2 block accessors (`attn_q/k/v/output_blocks_fp8e4m3`, `ffn_gate/up/down_blocks_fp8e4m3`, same for E5M2) and 14 Q4_0/Q8_0 block accessors added to `TransformerBlock`, delegating to the existing `LinearLayer::blocks_*` methods.
- **CUDA Q4_0 / Q8_0 GEMV kernels** (`oxibonsai-kernels`): `cuda_gemv_q4_0` and `cuda_gemv_q8_0` NVRTC kernels with warp-per-row reduction. Q4_0: 18-byte AoS blocks (FP16 scale + nibble-packed int4); Q8_0: 34-byte AoS blocks (FP16 scale + int8). Public functions callable from the forward dispatch in a future phase.

---

## [Unreleased] - Phase 20

### Added
- **CUDA ternary batch prefill** (`oxibonsai-kernels`): three new NVRTC kernels (`gemm_tq2_g128_v7`, `gemm_tq2_g128_v7_residual`, `fused_gate_up_swiglu_gemm_tq2`) enabling fused TQ2_0_g128 batch GEMM on CUDA. Replaces the sequential single-token fallback for Ternary models on CUDA hosts. Cap-of-8 outer loop prevents silent batch-size truncation (kernel_pattern_capof8). New `try_cuda_prefill_ternary` entry point; per-token `encode_attn_phase_tq2` for sequential attention within the batched prefill.
- **CUDA FP8 GEMV** (`oxibonsai-kernels`): `cuda_gemv_fp8_e4m3` and `cuda_gemv_fp8_e5m2` NVRTC kernels with warp-per-row warp-shuffle reduction and AoS block decode (32-weight blocks, FP16 scale at byte offset 32). `KernelDispatcher` with `KernelTier::Gpu` now routes `Fp8Kernel::gemv_fp8_e4m3/e5m2` calls to the GPU path on Linux/Windows.

---

## [Unreleased] - Phase 15 (0.1.5 preview)

### Added
- **FP8 quantization family** (`oxibonsai-core`): `BlockFP8E4M3` and `BlockFP8E5M2` block types (32 weights + FP16 scale = 34 bytes/block); bit-exact IEEE 754-style encode/decode with RNE rounding; GGUF type IDs 43/44 (PrismML FP8 extension); `is_fp8()` predicate in `GgufTensorType`; `F8_E4M3`/`F8_E5M2` forward-compat entries in `ExtendedQuantType` (8.5 bits/weight). Includes `fp8_e4m3_encode/decode` and `fp8_e5m2_encode/decode` public helpers.
- **FP8 reference kernels** (`oxibonsai-kernels`): scalar `dequant_fp8_e4m3/e5m2`, `gemv_fp8_e4m3/e5m2`, `gemm_fp8_e4m3/e5m2`; new `Fp8Kernel` trait mirroring `TernaryKernel`; `impl Fp8Kernel for KernelDispatcher` (all tiers currently route to scalar reference; SIMD specialization deferred to Phase 15.x).
- **`AllowListConstraint`** (`oxibonsai-runtime`): constrains output to a finite set of token-id sequences; candidate prefix tracking with activation bitmask; `active_count()` inspector. Useful for multiple-choice forced answers.
- **`SequenceConstraint`** (`oxibonsai-runtime`): forces output to follow a specific token-id sequence exactly; `is_failed()` inspector; returns `None` (unconstrained) once the full sequence is consumed.
- **`LengthConstraint`** (`oxibonsai-runtime`): hard `[min_len, max_len]` output length bounds with optional `stop_token`; excludes stop token before `min_len`, forces only stop token at `max_len`; `count()` inspector.
- **BNF grammar engine** (`oxibonsai-runtime`, new `grammar/` module): full context-free grammar support with an Earley recognizer (handles arbitrary CFG including left-recursive and ambiguous grammars via set-based memoization); hand-rolled BNF text parser supporting alternation, recursion, comments, line continuation, and escape sequences; `GrammarConstraint` implementing `TokenConstraint` for grammar-constrained token generation; pre-computed FIRST sets for O(1) next-byte lookahead; `next_byte_set()` API for direct byte-level constraint inspection. Pre-canned grammars: arithmetic, `a^n b^n`, CSV row, minimal JSON.

### Changed
- `GgufTensorType::from_id` now recognises IDs 43 (`F8_E4M3`) and 44 (`F8_E5M2`); tests updated to use ID 45 as the "unknown" probe.

---

## [0.1.4] - 2026-05-03

### Added
- `KvCacheCompressionPolicy` — runtime policy controller that adapts KV-cache precision (FP16 → Q8 → Q4) based on cache pressure thresholds, with EWMA-smoothed pressure tracking and explicit hysteresis to prevent thrashing
- `AdaptiveLookahead` — speculative-decoding draft length controller that updates the lookahead `k` from a running EWMA of accepted-tokens-per-step, clamped to a configurable `[min, max]` window
- `RequestRateTracker` — per-request EMA tokens/sec, p50/p95 inter-token latency, and queue-wait time tracking; surfaced in `InferenceMetrics` (`oxibonsai_request_tokens_per_second`, `oxibonsai_inter_token_latency_p50/p95_seconds`, `oxibonsai_queue_wait_seconds`)
- `RequestId` — UUIDv4-style 128-bit hex identifier with a deterministic xorshift64-based generator, addressable from new `oxibonsai_runtime::request_id` module for tracing-span correlation
- New Prometheus gauges in `InferenceMetrics`: `oxibonsai_request_tokens_per_second`, `oxibonsai_inter_token_latency_p50_seconds`, `oxibonsai_inter_token_latency_p95_seconds`, `oxibonsai_queue_wait_seconds`, `oxibonsai_kv_cache_compression_level`
- `InferenceMetrics::update_request_rate(&AggregateRateSnapshot)` and `InferenceMetrics::update_kv_cache_level(KvCacheLevel)` helpers
- `InferenceEngine::generate_tracked(&[u32], usize, &mut RequestRateTracker)` — generation that populates a tracker and pushes the snapshot to the engine's attached `RequestRateAggregator`
- `InferenceEngine::generate_with_request_id(RequestId, &[u32], usize) -> (Vec<u32>, RequestRateTracker)` — generation tagged with a UUIDv4 tracing span, returns the tracker for client-side telemetry
- `InferenceEngine::set_rate_aggregator(Arc<RequestRateAggregator>)` — workload-aggregator setter
- `examples/runtime_controllers.rs` — end-to-end demo of all four 0.1.4 controllers wired together
- `benches/controllers_bench.rs` — criterion microbenchmarks for `KvCachePolicy::observe`, `AdaptiveLookahead::observe_step`, `RequestRateTracker::record_token`, `RequestRateAggregator::record/snapshot`, and `RequestId::new/as_uuid/from_uuid`
- `tests/engine_controllers_tests.rs` — integration tests for the engine ↔ controller plumbing (8 tests)
- `GET /admin/workload-stats` — JSON endpoint exposing `RequestRateAggregator` snapshot (TBT p50/p95, EWMA tokens/sec, queue-wait, completed requests) and `KvCachePolicy` state (current tier, smoothed pressure, transition counters); both sources are optional on `AdminState`
- `AdminState::with_rate_aggregator(...)` and `AdminState::with_kv_cache_policy(...)` builder methods for attaching workload sources
- `RequestId::as_bytes() -> [u8; 16]` and `RequestId::from_bytes([u8; 16])` for binary-protocol round-trips (big-endian layout)
- `X-Request-ID` HTTP header support in the OpenAI server: client-supplied ids (UUID `8-4-4-4-12` form OR 32-char hex) are echoed back verbatim; absent or malformed headers trigger an auto-generated `RequestId`. Header constant `REQUEST_ID_HEADER` and helpers `resolve_request_id(&HeaderMap)` / `request_id_header_map(RequestId)` are public. Both streaming and non-streaming responses carry the header. Server tracing spans now record `request_id` for end-to-end correlation

### Changed
- Workspace version bump to 0.1.4 across all nine subcrates and `[workspace.dependencies]`
- `.github/workflows/ci.yml` re-enabled (was `.disabled`); branch list updated to track `main`, `master`, and any `0.1.*` line
- `SpeculativeDecoder` gained `with_adaptive(...)` constructor and an optional `AdaptiveLookahead` controller that updates the draft `k` after each step from the running acceptance EWMA
- `KvCachePolicyConfig` and `AdaptiveLookaheadConfig` no longer carry `#[non_exhaustive]` — these are simple value types and the attribute blocked struct-literal construction even with `..Default::default()`, hurting ergonomics in examples and benches
- Cleaned up three pre-existing `unused_variables` warnings in `oxibonsai-model::model::types::mod.rs` (the `gpu_kernel` binding is only consumed under specific feature combinations) using a `cfg_attr(...)`-gated `allow` so non-GPU builds remain warning-free

## [0.1.3] - 2026-05-03

### Added
- Prefix-cache–aware inference engine (`PrefixCachedEngine`) — KV-cache reuse across requests with cold/warm path parity
- Tokenizer auto-detection at runtime (sentencepiece vs. HF tokenizers) with a tokenizer bridge
- GPU weight cache management for `BonsaiModel` — single upload, reuse across decode steps
- CUDA ternary (TQ2) encoding kernels in `oxibonsai-kernels` (`encode_ternary` path) and full-forward CUDA fused layer
- Metal full-forward layer split into `forward_metal.rs` / `forward_cuda.rs` modules with shared `gpu_cache.rs`
- `oxibonsai-tokenizer` upgrades and CLI tokenizer management commands

### Changed
- Workspace version bump to 0.1.3 across all nine subcrates and `[workspace.dependencies]`
- Upgraded `oxifft` workspace dependency to 0.3
- Updated CUDA dependencies (`cudarc`) for better CUDA 11.x / 12.x compatibility
- `oxibonsai-runtime` server and engine refactored to thread the prefix-cache and tokenizer-bridge plumbing
- `download_ternary.sh` script: parallel downloads and improved error messages
- Refactored `oxibonsai-model::model::types` (1857 lines) into a `types/` directory with `mod.rs`, `forward_cuda.rs`, `forward_metal.rs`, `gpu_cache.rs` (under-2000-lines policy)

### Fixed
- `PrefixCachedEngine` cached-path output now matches cold-cache path byte-for-byte in tests
- LF line-ending enforcement for shell scripts via `.gitattributes`

## [0.1.2] - 2026-04-19

### Added
- ONNX MatMulNBits (bits=2) ingestion — `oxibonsai convert --onnx` reads onnx-community Ternary releases directly and repacks them as GGUF (TQ2_0_g128)
- Qwen3 ONNX tensor role mapping for the converter

### Changed
- Upgraded `oxionnx-proto` workspace dependency to 0.1.2
- Workspace version bump to 0.1.2 across all nine subcrates and `[workspace.dependencies]`
- Alpha → Stable uplift for `oxibonsai-tokenizer`, `oxibonsai-rag`, `oxibonsai-eval`, and `oxibonsai-serve`

## [0.1.1] - 2026-04-18

### Added
- Native CUDA NVRTC backend with fused Q1 + TQ2 full-forward path (~21.9 tok/s on Ternary-Bonsai-1.7B : RTX 3060 CUDA 12.8)
- Fused Metal TQ2 full-forward — single GPU command buffer per token, ~50 tok/s on Ternary-Bonsai-1.7B (~13× speedup)
- Ternary CPU SIMD tiers (NEON/AVX2/AVX-512 TQ2 GEMV)
- TQ2_0_g128 support in the Metal backend (per-kernel dispatch + `blocks_as_bytes_ternary` zero-copy upload)
- `scripts/bench_ternary.sh` — CPU vs Metal throughput bench (3-run average + best)
- `scripts/download_ternary.sh` — fetch + convert safetensors → GGUF

### Changed
- Version bump to 0.1.1
- Internal dependency version alignment across workspace
- CUDA full-forward layer parameter handling refactored for cleaner weight management
- Workspace Cargo.toml files unified on workspace dependencies for better crate compatibility

### Fixed
- Workspace version consistency across all subcrates
- `blocks_as_bytes` import gating for broader feature-flag compatibility

## [0.1.0] - 2026-04-13

### Added

- Pure Rust 1-bit LLM inference engine for PrismML Bonsai models
- GGUF Q1_0_g128 format support with streaming parser
- Optimized 1-bit kernel operations (dequantization, GEMV, GEMM)
- SIMD acceleration support (AVX2, AVX-512, NEON, WASM SIMD)
- Parallel kernel dispatch with Rayon
- Qwen3 transformer model implementation with paged KV-cache
- High-level inference runtime with autoregressive generation
- Sampling strategies (greedy, top-k, top-p, temperature)
- OpenAI-compatible REST API server (chat completions, completions, embeddings)
- Streaming token generation via SSE
- RAG pipeline with chunking and similarity search
- Pure Rust BPE tokenizer
- Model evaluation framework (accuracy, perplexity metrics)
- WASM compilation target support
- Speculative decoding support
- Comprehensive test suite (140 tests)
- Cross-platform support (macOS, Linux, Windows, WASM)

[0.1.2]: https://github.com/cool-japan/oxibonsai/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/cool-japan/oxibonsai/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/cool-japan/oxibonsai/releases/tag/v0.1.0
