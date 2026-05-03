# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
