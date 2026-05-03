# OxiBonsai TODO

> Pure Rust 1-bit LLM inference engine for PrismML Bonsai models
> 412 source files, ~152,615 lines of Rust code, 3,589+ tests passing across workspace — verified 2026-05-03 (0.1.4)

## Phase Status

| Phase | Status | Summary |
|-------|--------|---------|
| Phase 0: Foundation | ✅ Complete | Workspace, GGUF loader, metadata extraction |
| Phase 1: 1-Bit Kernels | ✅ Complete | dequant/gemv/gemm + property tests (20 kernel tests) |
| Phase 2: Transformer Engine | ✅ Complete | RMSNorm, SwiGLU, RoPE, GQA, 36-layer Qwen3 forward pass |
| Phase 3: Inference Runtime | ✅ Complete | KV cache, sampling, tokenizer, CLI (run/chat/serve/info), OpenAI-compatible server |
| Phase 4: Production Hardening | ✅ Complete | AVX2 SIMD, NEON, AVX-512, rayon, benchmarks, tracing, TOML config, proptests, integration tests, circuit breaker, health, metrics, recovery, Prometheus metrics, memory profiling, fuzz testing |
| Phase 5: Ecosystem Integration | ✅ Complete | All major subsystems implemented (see below) |
| Phase 6: Advanced Infrastructure | ✅ Complete | Multi-GPU, CUDA/Metal stubs, PagedAttention, GGUF v3, dynamic quant, profiler |
| Phase 7: Production Features | ✅ Complete | Model merging, checkpointing, flash decoding, attention sink, compression, ROUGE, MoD, adaptive sampling, oxibonsai-serve |
| Phase 8: Final Polish | ✅ Complete | K-quant formats, streaming GGUF, kernel tuning, comprehensive testing |
| Phase 9: Ternary Bonsai | ✅ Complete | TQ2_0_g128 block type, SIMD kernels, model variants, GGUF surface, export |
| Phase 10: Ternary CPU SIMD tiers | ✅ Complete | AVX2 / AVX-512 / NEON TQ2 GEMV with pos_mask+neg_mask decomposition |
| Phase 11: Metal TQ2 GEMV + dispatch | ✅ Complete | Per-kernel Metal TQ2_0_g128 dispatch in `gpu_backend/metal_*` |
| Phase 12: Native CUDA backend | ✅ Complete | NVRTC kernels, fused Q1 + TQ2 full-forward; ~21.9 tok/s on Ternary-Bonsai-1.7B |
| Phase 13.x: Fused Metal TQ2 full-forward | ✅ Complete | Single command buffer per token, ~13× speedup; ~50 tok/s on Ternary-Bonsai-1.7B |
| Phase 13.y: Ternary LM head on GPU | ✅ Complete | All 7 `OutputWeight::Ternary` guard sites closed (4 Metal + 3 CUDA); Metal ~54.5 tok/s avg on Ternary-Bonsai-1.7B |
| Phase 14: Runtime controllers (0.1.4) | ✅ Complete | `KvCachePolicy`, `AdaptiveLookahead`, `RequestRateTracker`, `RequestId`, expanded Prometheus surface |

## Phase 4 — Complete

- [x] **NEON (ARM) SIMD kernels** -- Implemented `simd_neon.rs` with dequant/gemv/gemm
- [x] **AVX-512 SIMD kernels** -- Implemented `simd_avx512.rs` (compile-checked, untested on ARM)
- [x] **Property-based tests** -- proptest roundtrip, linearity, cross-tier in `proptest_kernels.rs`
- [x] **Integration tests** -- `tests/integration_pipeline.rs` + `tests/cross_tier.rs` + `tests/feature_flags.rs`
- [x] **Tracing enhancement** -- EnvFilter + `#[instrument]` spans on generate, forward, block, server, sampling
- [x] **TOML configuration system** -- `OxiBonsaiConfig` with layered loading, `--config` CLI arg
- [x] **Circuit breaker** -- Overload protection with configurable thresholds
- [x] **Health checks** -- `/health` endpoint with readiness probes
- [x] **Inference metrics** -- tokens/sec, latency, request counts via `InferenceMetrics`
- [x] **Builder pattern API** -- `EngineBuilder`, `ConfigBuilder`, `SamplerBuilder`
- [x] **Sampling presets** -- Greedy, Balanced, Creative, Code presets
- [x] **Batch engine** -- Batched inference support
- [x] **Async engine** -- Tokio-based async inference
- [x] **Recovery strategies** -- Error classification and automatic recovery
- [x] **Prometheus metrics** -- Self-contained Prometheus-compatible metrics; `/metrics` endpoint via `InferenceMetrics` + `metrics_text()` renderer
- [x] **Memory profiling** -- `MemoryProfiler` + `MemorySnapshot`; RSS via Mach (macOS) / `/proc/self/statm` (Linux); peak tracking
- [x] **Fuzz testing** -- Property-based kernel + GGUF parsing fuzz tests; cargo-fuzz harnesses being added

## Phase 5 — Complete

- [x] **Streaming chat completions (SSE)** -- Implemented with `generate_streaming` + SSE handler
- [x] **WASM compilation target** -- Compiles to `wasm32-unknown-unknown`; `wasm_api.rs` provides WASM-safe generate API
- [x] **Bonsai-4B and Bonsai-1.7B model support** -- `Qwen3Config::bonsai_4b()` / `bonsai_1_7b()`, `ModelVariant::Bonsai4B` / `Bonsai1_7B`, `model_variants.rs` with full `ModelSpec` + `CapabilityProfile`
- [x] **OxiRAG integration** -- Pure Rust RAG pipeline (`oxibonsai-rag`): chunker, IdentityEmbedder, TfIdfEmbedder, VectorStore, Retriever, RagPipeline
- [x] **MeCrab/OxiTokenizer** -- Native BPE tokenizer stub (`oxibonsai-tokenizer`): Vocabulary, BpeMerges, OxiTokenizer, char-level fallback, ChatTemplate/chatml
- [x] **Advanced sampling** -- Mirostat v1/v2, Locally Typical, Eta, Min-P samplers; `SamplerChain` composable pipeline; `LcgRng`
- [x] **Speculative decoding** -- `SpeculativeDecoder` with draft/verify loop, acceptance rate tracking, `generate_speculative`
- [x] **LoRA adapter support** -- `LoraAdapter`, `LoraRegistry`, `BonsaiLoraSet`; apply/merge into weights
- [x] **Context window management** -- `ContextWindow` with TruncateLeft/Right/SlidingWindow/Summarize strategies; system-prompt preservation
- [x] **Beam search** -- `BeamSearchEngine` with configurable `beam_width`, length penalty, no-repeat n-gram, early stopping
- [x] **Token healing** -- `TokenHealer`, `TokenHealingConfig`, `HealingDecoder`
- [x] **Inference pipeline API** -- `InferencePipeline`, `PipelineBuilder`, `GenerationStrategy`, `StopReason`, `PipelineOutput`; convenience constructors (`chat_pipeline`, `code_pipeline`, `greedy_pipeline`)
- [x] **INT8 quantization** -- Per-tensor and per-channel INT8 quantization; `Int8Tensor` with `dequantize()` and `matvec()`
- [x] **GGUF writer** -- `GgufWriter` producing valid GGUF byte streams with metadata and tensor entries
- [x] **Model export** -- `export_to_gguf` supporting Float32, Q1_0_g128, Int8PerChannel formats; FP32 exception layers; `ExportStats`
- [x] **Prefix caching** -- `PrefixCache`, `CacheBlock`, `CacheSession`, `PrefixAwarePrefill` for KV reuse
- [x] **Semantic cache** -- `SemanticCache` with cosine-similarity lookup for repeated prompts
- [x] **MoE architecture** -- Mixture-of-Experts router and expert networks (`moe_router.rs`, `moe_expert.rs`)
- [x] **Tensor parallelism** -- `TensorParallelConfig`, column/row parallel linear; `AllReduceOp`
- [x] **Grammar-constrained decoding** -- `TokenConstraint` trait; `JsonConstraint`, `RegexConstraint`, `AllowListConstraint`, `SequenceConstraint`
- [x] **OpenAI API extensions** -- Function calling, logprobs, JSON mode, `n` completions, response format, frequency/presence penalties (`api_extensions.rs`, `api_types.rs`)
- [x] **Embeddings endpoint** -- `POST /v1/embeddings`; `EmbedderRegistry` (IdentityEmbedder + TfIdfEmbedder); float + base64 encoding; dimensions truncation
- [x] **Completions endpoint** -- `POST /v1/completions` with single/batch prompt, echo, logprobs stub
- [x] **RAG server endpoints** -- `/v1/rag/index`, `/v1/rag/query` (feature-gated)
- [x] **Rate limiting + middleware** -- `RateLimiter`, `MiddlewareConfig`; tower-based rate-limit and timeout layers
- [x] **Admin API** -- `/admin/status`, `/admin/config`, `/admin/reset-metrics`, `/admin/cache-stats`
- [x] **Evaluation harness** -- `oxibonsai-eval` crate: HellaSwag, MMLU, TruthfulQA benchmarks; `EvalRunner`, `EvalResult`
- [x] **Optimizer suite** -- SGD (with momentum, Nesterov, weight decay), Adam, AdamW; CosineAnnealing / WarmupConstant / Constant LR schedulers
- [x] **LoRA training scaffold** -- `LoraTrainer` with per-step Adam updates, gradient clipping, warmup; `TrainingStep` metrics
- [x] **Fine-grained gradient computation** -- `GradientContext`, `LayerGradients`, `backward_rms_norm`, `backward_swiglu`, `backward_linear`
- [x] **Continuous batching** -- `ContinuousBatchScheduler` for in-flight request management
- [x] **Model registry** -- `ModelVariant`, `ModelSpec`, `CapabilityProfile`; auto-detection from config dimensions
- [x] **Comprehensive integration tests** -- `tests/full_pipeline_tests.rs` (25 tests), `tests/quantization_pipeline_tests.rs` (10 tests), `tests/server_integration_tests.rs` (8 tests)

## CI / Publish

- [x] Publish script (`scripts/publish.sh`) -- dry-run by default, `--for-real` for actual publish
- [x] Feature flag tests (`tests/feature_flags.rs`) -- compile-time validation of feature propagation
- [x] GitHub Actions CI pipeline -- `.github/workflows/ci.yml` (build, test, clippy, WASM, feature matrix)
- [ ] crates.io publish (awaiting approval)

## Phase 6 — Complete

- [x] **True multi-GPU inference** -- NCCL-style all-reduce/gather/scatter; `DeviceMesh` (tp×pp); rayon-simulated collectives; column/row weight partition + merge helpers; `multi_gpu.rs` + 27 tests
- [x] **CUDA/Metal backend** -- GPU kernel dispatch stubs: `GpuBackend` trait, `CpuBackend` (always-on), `CudaBackend` (feature = "cuda"), `MetalBackend` (feature = "metal"); `select_backend`, `gpu_matmul`; 18 integration tests; cuBLAS / Metal Performance Shaders left as future implementation
- [x] **Production GGUF with real model weights** -- Streaming `TensorChunkIter`; memory budget estimation (`estimate_memory_bytes`, `fits_in_budget`); lazy tensor metadata loading; `validate_gguf_file`; `gguf_loader.rs` + 25 tests
- [x] **Web UI** -- Chat interface; model management dashboard
- [x] **Distributed serving** -- Multi-node coordinator; consistent hashing for request routing
- [x] **Full MeCrab tokenizer integration** -- `NativeTokenizerBridge` added to `oxibonsai-runtime`; wraps `OxiTokenizer` with encode/decode/chat-format API; pure Rust, WASM-compatible, zero C/FFI; 10 integration tests all passing
- [x] **PagedAttention / vLLM-style KV management** -- Block-based KV cache for higher utilisation; `PagedKvCache`, `BlockPool`, `BlockTable`, `KvPage` in `oxibonsai-model/src/paged_kv_cache.rs`
- [x] **GGMLv4 / GGUF v3 support** -- Forward compat layer: `GgufVersion`, `ExtendedQuantType`, `GgufCompatReport`, `check_gguf_header`, `build_compat_report` in `gguf/compat.rs`; 20 integration tests all passing
- [x] **Sparse attention patterns** -- `SparseAttentionMask` with Local Window, Strided, Bigbird, Longformer, and Dilated patterns; `sparse_attention_forward`; memory reduction estimate; `sparse_attention.rs` + tests in `oxibonsai-model`
- [x] **Weight tying** -- `TiedEmbedding` with tied embedding/unembedding pair; `embed()`, `lm_head_logits()`, `set_embedding()` API; `TyingError` type; `weight_tying.rs` + tests in `oxibonsai-model`
- [x] **Dynamic quantization** -- `DynamicQuant` trait; `DynamicQ8_0`, `DynamicQ4_0`, `DynamicQ4_1` quantizers; `QuantizedTensor`, `quantize_model_dynamic`, `calibrate_scale`; `dynamic_quant.rs` + 21 tests
- [x] **Runtime profiler** -- `InferenceProfiler` with span tracking, `SpanRecord`, `ProfileReport`, stats (mean/min/max/p95/p99); flame-graph text renderer; `profiler.rs` + 20 tests
- [x] **Advanced RAG chunking** -- `SemanticChunker` with cosine-similarity boundary detection; `HierarchicalChunker` with multi-level chunking; `SlideWindowChunker`; 31 tests in `oxibonsai-rag`
- [x] **Tokenizer serialization** -- `save_json`/`load_json` for `OxiTokenizer`; vocabulary and BPE merge round-trip; `TokenizerSerializer`; 16 tests in `oxibonsai-tokenizer`
- [x] **GGUF model card** -- `ModelCard` with metadata (author, license, description, tags, datasets); `CardFormat` enum; `render_card`/`parse_card`; embedded in GGUF metadata via `oxibonsai-core`
- [x] **Token streaming metrics** -- `StreamingSession` with per-token timing; `TokenEvent`, `StreamingStats` (mean/p50/p95 TTFT + TPOT); `StreamingProfiler`; 22 tests in `oxibonsai-runtime`
- [x] **KV cache quantization** -- `QuantizedKvCache` with Q8 and Q4 formats; per-head scale factors; `KvCacheQuantizer`, `store_quantized`, `load_dequantized`; `kv_cache_quant.rs` + 18 tests

## Phase 7 — Complete

- [x] **KV cache quantization (INT8)** -- `QuantizedKvCache` with Q8/Q4 formats; per-head scale factors; `kv_cache_quant.rs`
- [x] **Model merging (SLERP, TIES, DARE, task vector)** -- `model_merge.rs` with multi-strategy merge support
- [x] **Checkpoint save/load (OXCK binary format)** -- `checkpoint.rs` with serialization/deserialization
- [x] **YaRN extended RoPE** -- `yarn_rope.rs` with YaRN-style RoPE scaling for extended context
- [x] **Flash Decoding** -- `flash_decode.rs` for optimized single-token decoding
- [x] **Attention Sink / StreamingLLM** -- `attention_sink.rs` with sink tokens for infinite context windows
- [x] **Dynamic activation quantization** -- `dynamic_quant.rs` with DynamicQ8_0/Q4_0/Q4_1 quantizers
- [x] **Inference profiler with FLOP counting** -- `profiler.rs` with span tracking, p95/p99 stats, flame-graph renderer
- [x] **GGUF model card generation** -- `ModelCard` with metadata; `render_card`/`parse_card` in `oxibonsai-core`
- [x] **Token streaming metrics (TTFT, TBT)** -- `StreamingSession`, `StreamingStats`, `StreamingProfiler` in `oxibonsai-runtime`
- [x] **Weight importance analysis and pruning** -- `pruning.rs` with structured/unstructured pruning
- [x] **PTQ calibration pipeline** -- `calibration.rs` with post-training quantization calibration
- [x] **BPE tokenizer training** -- `trainer.rs` in `oxibonsai-tokenizer` with BPE merge learning
- [x] **Tokenizer serialization** -- `save_json`/`load_json` for `OxiTokenizer`; vocabulary and BPE merge round-trip
- [x] **Advanced RAG chunking** -- Sentence, recursive, sliding window, markdown chunking in `oxibonsai-rag`
- [x] **Loss functions (CE, focal, distillation, contrastive)** -- `losses.rs` with full training loss suite
- [x] **Advanced LR schedulers (OneCycle, ReducePlateau, CyclicLR)** -- `lr_schedulers.rs` with advanced scheduling
- [x] **Sparse attention patterns (local window, BigBird, strided)** -- `sparse_attention.rs` with multiple patterns
- [x] **Weight tying (embedding/LM head sharing)** -- `weight_tying.rs` with `TiedEmbedding`
- [x] **Model compression pipeline (prune+quantize)** -- `compression.rs` with end-to-end compression
- [x] **ROUGE evaluation metrics (ROUGE-1/2/L/S)** -- `rouge.rs` in `oxibonsai-eval`
- [x] **Mixture of Depths (adaptive compute)** -- `mixture_of_depths.rs` with `ModRouter`
- [x] **Token budget tracking** -- `token_budget.rs` in `oxibonsai-runtime`
- [x] **RoPE scaling variants (linear, DynamicNTK, LLaMA 3.1, LongRoPE)** -- `rope_scaling.rs` with multiple strategies
- [x] **N-best hypothesis tracking** -- `nbest.rs` in `oxibonsai-runtime`
- [x] **Adaptive sampling (entropy cooling, repetition adaptation)** -- `adaptive_sampling.rs` with `AdaptiveChain`
- [x] **Generation quality metrics (repetition, diversity, BLEU)** -- `quality_metrics.rs` in `oxibonsai-runtime`
- [x] **oxibonsai-serve standalone binary crate** -- `crates/oxibonsai-serve` with CLI args, banner, OpenAI-compatible server
- [x] **Cross-attention layers** -- `cross_attention.rs` for encoder-decoder architectures
- [x] **Gradient checkpointing** -- `gradient_checkpoint.rs` for memory-efficient training
- [x] **Request deduplication** -- `dedup.rs` with FNV-1a hashing for duplicate request detection
- [x] **Model hot-reload** -- `hot_reload.rs` for live model swapping without restart
- [x] **Performance auto-tuning engine** -- `auto_tuner.rs` with CPU detection, SIMD tier selection, memory budget estimation, kernel benchmarking
- [x] **JSON Schema structured output constraint** -- `json_schema.rs` with schema parsing, validation state machine, constrained token generation
- [x] **Multi-model serving with LoRA routing** -- `multi_model.rs` / `lora_router` with dynamic model selection and LoRA adapter dispatch

## Phase 8 — Complete

- [x] **Q2_K and Q4_K quantization formats** — `BlockQ2K`/`BlockQ4K` with dequant/quantize in `oxibonsai-core/quant_k.rs`; 21 tests
- [x] **Streaming GGUF reader** — `GgufStreamParser` state machine for progressive network-loaded GGUF parsing; 22 tests
- [x] **Kernel performance tuning** — `PlatformProfile` detection, `TunedThresholds` auto-computation, cache-line aligned buffers, software prefetch hints; 29 tests
- [x] **Layer-level correctness tests** — RMSNorm, SwiGLU, RoPE, Attention, TransformerBlock reference comparisons; 23 tests
- [x] **Numerical stability tests** — Extreme inputs, overflow/underflow, long sequences, KV cache stress; 25 tests
- [x] **Full generate() pipeline integration tests** — End-to-end generation, determinism, sampling parameter effects, edge cases; 26 tests
- [x] **Sampling distribution statistical tests** — Chi-square goodness-of-fit, temperature/top-k/top-p/repetition penalty validation; 22 tests

## Phase 9 — Ternary Bonsai

- [x] **Quant block types** — `crates/oxibonsai-core/src/quant_ternary.rs`: `BlockTQ2_0_g128` (34B/128w), `BlockTQ2_0` (66B/256w), `TernaryCode`, scalar dequant/quantize, zero-copy `slice_from_bytes`, 11 tests
- [x] **GGUF surface** — `GgufTensorType::TQ2_0=37/TQ2_0_g128=42`, `writer::TensorType`, `KNOWN_QUANT_TYPES`, `compute_tensor_size_bytes`, `is_ternary()` helper, ID verification against ggml-quants.h, 6 tests
- [x] **Core re-exports** — `pub mod quant_ternary` + re-export `BlockTQ2_0{_g128}`, `TernaryCode`, constants from `oxibonsai_core`
- [x] **Scalar ternary kernels** — `dequant_ternary.rs`, `gemv_ternary.rs`, `gemm_ternary.rs` correctness reference; 9 tests per file
- [x] **SIMD ternary kernels** — NEON / AVX2 / AVX-512 `gemv_tq2_0_g128` + `dequant` + `gemm`; pos_mask+neg_mask decomposition; sign-canary tests
- [x] **`TernaryKernel` trait + dispatcher** — `traits.rs` + `dispatch.rs` impl; GPU tier falls back to scalar; 2 dispatcher tests
- [x] **`Qwen3Config::ternary_bonsai_*()` constructors** — three named constructors in `config.rs`, co-drift via 1-bit ctor call; 2 tests
- [x] **`ModelVariant::TernaryBonsai*` + detection** — three enum variants + `from_config_and_sample_tensor_type()`; all-ternary byte budget; 4 tests
- [x] **`ternary_bonsai_*_spec()` + capability profiles** — six-entry `all_specs()`, ±5 % byte budget tests; 4 tests
- [x] **`weight_loaders.rs` + `LinearTernary` layer** — `load_ternary_blocks`, `load_ternary_embedding`, `OutputWeight::Ternary`; sum-type for Attn/MLP; 3 tests
- [x] **`ExportFormat::TernaryG128`** — `quantize_ternary.rs`, norm-only FP32 exceptions, `default_fp32_exceptions_ternary_excludes_embedding` guard; 5 tests
- [x] **Phase 9 docs** — TODO.md Phase Status table + Memory/Performance rows + crate-local entries; README.md Ternary Bonsai subsection

## Phase 10 — Complete

- [x] **NEON TQ2 GEMV** — `simd_neon.rs` ternary GEMV with pos_mask + neg_mask decomposition over TQ2_0_g128 blocks
- [x] **AVX2 TQ2 GEMV** — `simd_avx2.rs` ternary GEMV over 128-weight groups with FP16 scale broadcast
- [x] **AVX-512 TQ2 GEMV** — `simd_avx512.rs` ternary GEMV with 512-bit lanes
- [x] **Ternary dispatch** — `TernaryKernel` routes via `KernelDispatcher::auto_detect()` across NEON / AVX2 / AVX-512 / scalar tiers
- [x] **Ternary dequant SIMD** — per-tier `dequant_ternary` for TQ2_0_g128 (mirrors the Q1 dequant layout)
- [x] **Apple Silicon CPU baseline** — NEON TQ2 path measured at ~7–8 tok/s end-to-end on Ternary-Bonsai-1.7B via `scripts/cli_ternary.sh`

## Phase 11 — Complete

- [x] **Metal TQ2_0_g128 GEMV kernel** — per-kernel TQ2 dispatch path in `crates/oxibonsai-kernels/src/gpu_backend/metal_dispatch.rs`
- [x] **`blocks_as_bytes_ternary` helper** — zero-copy TQ2 block upload to the Metal backend
- [x] **Per-kernel dispatch parity with Q1** — Metal TQ2 path validated against scalar reference so CPU and Metal produce byte-identical output at `--temperature 0 --seed 42`
- [x] **Metal graph groundwork** — `gpu_backend/metal_graph.rs` and `metal_prefill.rs` cover prefill + per-step GEMV composition ahead of the Phase 13.x fused path

## Phase 12 — Complete

- [x] **NVRTC kernel pipeline** — `gpu_backend/cuda_kernels.rs`, `cuda_attn_kernels.rs`, `cuda_prefill_kernels.rs`; runtime-compiled PTX in `kernel_sources/`
- [x] **CUDA graph capture** — `gpu_backend/cuda_graph/` groups every layer op (RMSNorm, gemv v7/v8/v9 with residual, SwiGLU, gate+up, Q1 AoS→SoA and TQ2 block→SoA reformat) into a reusable CUDA graph
- [x] **Fused Q1 + TQ2 full-forward** — `cuda_full_layer.rs` encodes a full Qwen3 layer into one graph launch for both quant families
- [x] **Prefill path** — `gpu_backend/cuda_prefill.rs` shares kernel sources with decode to keep the prefill/decode invariants identical
- [x] **`native-cuda` feature flag** — opt-in backend selected via `KernelDispatcher` alongside `cuda` (scirs2) and `metal`
- [x] **Measured** — ~21.9 tok/s on Ternary-Bonsai-1.7B end-to-end via `scripts/cli_ternary.sh cuda`

## Phase 13.x — Complete

- [x] **Fused Metal TQ2 full-forward** — `gpu_backend/metal_full_layer/` encodes the 14-dispatch sequence (pre-attn RMSNorm → fused QKV → QK-norm + RoPE → KV-store → batched attention → attn output + residual → FFN RMSNorm → fused gate+up → SwiGLU → down + residual) into a single GPU command buffer per token
- [x] **Fused Metal Q1 full-forward** — same command-buffer layout reused for 1-bit weights; shared with the TQ2 code path via a weight-family-generic encoder
- [x] **Command buffer reuse** — per-layer argument buffers cached in `metal_full_layer/types.rs` to avoid per-token allocations
- [x] **`scripts/bench_ternary.sh`** — 3-run CPU-vs-Metal averaging harness with best-of-N reporting
- [x] **`scripts/download_ternary.sh`** — HF `hf` CLI fetch + safetensors → GGUF conversion wrapper
- [x] **Measured** — ~50 tok/s (best ~57) on Ternary-Bonsai-1.7B on Apple Silicon (M-series), ~13× speedup vs the per-GEMV dispatch of Phase 11

## Phase 13.y — Complete (shipped in 0.1.3)

Closes all 7 `OutputWeight::Ternary(_)` CPU-fallback guard sites in `crates/oxibonsai-model/src/model/types.rs`.

### Milestone 1 — Metal ternary LM head (4 guard sites)

- [x] Pre-split `model/types.rs` (1857 lines → >2000 after M1 additions) via `splitrs` (completed 2026-04-24)
  - **Goal:** Keep every file under 2000 lines. Split into `types.rs` + `types_gpu_cache.rs` + `types_forward_metal.rs`.
  - **Files:** `crates/oxibonsai-model/src/model/types/mod.rs` (541), `gpu_cache.rs` (233), `forward_metal.rs` (1054), `forward_cuda.rs` (454).
  - **Tests:** `cargo nextest run -p oxibonsai-model --all-features` green after split.

- [x] Add `MetalGraph::encode_tail_and_commit_ternary` + update `encode_full_forward_ternary` routing + update `try_metal_full_forward_ternary` LM-head upload (completed 2026-04-24)
  - **Goal:** Kernel-layer ternary LM head support. `encode_tail_and_commit_ternary` mirrors `encode_tail_and_commit` but uses `dispatch_gemv_tq2` at the LM-head GEMV site. `encode_full_forward_ternary` routes to it. `try_metal_full_forward_ternary` uploads via `get_or_upload_tq2_weight_soa` with namespace `4_000_000+`.
  - **Files:** `crates/oxibonsai-kernels/src/gpu_backend/metal_full_layer/functions_2.rs`, `functions_3.rs`, `metal_graph.rs` (test).
  - **Tests:** `test_encode_tail_ternary_matches_reference` passes — bit-exact token match, logits within 1e-3.

- [x] Extend `CachedModelWeights` + remove short-circuit + wire ternary cache builder + remove 4 Metal guards (completed 2026-04-24)
  - **Goal:** Model-layer: `CachedModelWeights` extended with ternary fields; ternary cache builder wired; all 4 Metal Err guards replaced with `try_metal_*_ternary` routes; 3 new thin wrappers in `functions_3.rs`.
  - **Files:** `crates/oxibonsai-model/src/model/types/gpu_cache.rs`, `forward_metal.rs`; `crates/oxibonsai-kernels/src/gpu_backend/metal_full_layer/functions_3.rs`.
  - **Tests:** CPU vs Metal greedy-decode output byte-matches on Ternary-Bonsai-1.7B (temp=0, seed=42). Metal avg=54.5 tok/s (best=55.4) — improvement from ~50 tok/s baseline.

### Milestone 2 — CUDA ternary full-forward (3 guard sites)

- [x] Pre-split `cuda_full_layer.rs` (1687 lines → >2000 after M2 additions) via `splitrs` (completed 2026-04-24)
  - **Goal:** Split into `cuda_full_layer/mod.rs` + `encode_q1.rs` + `encode_ternary.rs` (new, M2 target).
  - **Files:** `crates/oxibonsai-kernels/src/gpu_backend/cuda_full_layer.rs` → directory.
  - **Tests:** `cargo check --features cuda` green (execution gated by CI-GPU).

- [x] Implement CUDA ternary full-forward path (completed 2026-04-24)
  - **Goal:** `encode_layer_into_ternary` (14 dispatches/layer, same split-dispatch shape as Metal Phase 13.x), `encode_full_forward_ternary`, `encode_lm_head_gemv_ternary` (single `launch_gemv_tq2_v1`), `try_cuda_full_forward_ternary` + `try_cuda_full_forward_ternary_with_gpu_lm_head`. Remove 3 CUDA guards at `try_cuda_full_forward_with_lm_head`, `try_cuda_prefill_with_lm_head`, `try_cuda_prefill_verify`.
  - **Files:** `crates/oxibonsai-kernels/src/gpu_backend/cuda_full_layer/encode_ternary.rs` (implemented); `crates/oxibonsai-model/src/model/types/forward_cuda.rs` (3 guards removed).
  - **Tests:** `test_encode_lm_head_gemv_ternary_matches_reference`, `test_encode_full_forward_ternary_matches_reference` (CI-GPU-gated). `cargo check --all-features` + clippy + 140 non-CUDA tests all green.
  - **Risk:** `CuGraphHolder` captured-graph pattern — capture+replay must be tested end-to-end, not just isolated kernel launches. Post-merge validation on CUDA hardware required.

## Phase 14 — Runtime Controllers (0.1.4)

- [x] **`KvCachePolicy`** — runtime KV-cache compression policy: FP16 → Q8 → Q4 transitions driven by EWMA pressure with hysteresis to prevent thrashing; configurable thresholds and `[min, max]` clamps; thread-safe via atomics
- [x] **`AdaptiveLookahead`** — speculative-decoding draft-length controller; updates `k` from a running EWMA of accepted tokens per step; cooldown to suppress oscillation; clamped to a configured `[min, max]`
- [x] **`RequestRateTracker` + `RequestRateAggregator`** — per-request EWMA tokens/sec, p50/p95 inter-token latency (TBT), queue-wait time; workload rollup aggregator with sliding-window ring buffer
- [x] **`RequestId`** — RFC 4122 UUIDv4-style 128-bit identifiers with thread-safe SplitMix64 generator (no external dependencies); hex / UUID / display formatting; round-trip parse
- [x] **Prometheus surface expansion** — five new gauges: `oxibonsai_request_tokens_per_second`, `oxibonsai_inter_token_latency_p50/p95_seconds`, `oxibonsai_queue_wait_seconds`, `oxibonsai_kv_cache_compression_level`
- [x] **`InferenceMetrics::update_request_rate(...)` and `update_kv_cache_level(...)` helpers** — wire the workload aggregator + KV policy directly into Prometheus output
- [x] **`SpeculativeDecoder::with_adaptive(...)`** — wires `AdaptiveLookahead` into the speculative loop; lookahead updates after every step
- [x] **CI re-enabled** — `.github/workflows/ci.yml` restored from `.disabled`; branches updated to `main` / `master` / `0.1.*`

## Performance Targets

| Platform | Target | Status |
|----------|--------|--------|
| x86-64 AVX2 (Zen4) | >= 20 t/s | 🔶 Kernels done, not benchmarked end-to-end |
| x86-64 AVX-512 | >= 30 t/s | 🔶 Kernels done, not benchmarked end-to-end |
| ARM NEON | >= 18 t/s | 🔶 Kernels done, not benchmarked end-to-end |
| WASM | >= 5 t/s | 🔶 Compiles; runtime throughput not benchmarked |
| ARM NEON (ternary) | >= 14 t/s | 🔶 Apple Silicon measured ~7–8 t/s end-to-end on Ternary-Bonsai-1.7B |
| x86-64 AVX2 (ternary) | >= 18 t/s | 🔶 Kernels done, not benchmarked end-to-end |
| x86-64 AVX-512 (ternary) | >= 26 t/s | 🔶 Kernels done, not benchmarked end-to-end |
| Metal fused TQ2 (Apple Silicon) | >= 30 t/s | ✅ Measured avg=54.5 t/s (best=55.4) on Ternary-Bonsai-1.7B (Phase 13.y, 2026-04-24; was ~50 t/s after Phase 13.x) |
| CUDA native TQ2 (NVIDIA GPU) | >= 20 t/s | ✅ Measured ~21.9 t/s on Ternary-Bonsai-1.7B |

## Memory Targets

| Metric | Target | Status |
|--------|--------|--------|
| Model weights (mmap) | 1.15 GB | ✅ mmap zero-copy implemented |
| KV cache (4k context) | ~256 MB | ✅ Implemented |
| Runtime overhead | < 50 MB | 🔶 Profiler available via `MemoryProfiler`; end-to-end not yet characterized |
| Total (4k context) | < 1.5 GB | 🔶 Not profiled end-to-end |
| Ternary-Bonsai-8B | ~1.75 GB | TQ2_0_g128 (34B/128w) |
| Ternary-Bonsai-4B | ~900 MB | TQ2_0_g128 (34B/128w) |
| Ternary-Bonsai-1.7B | ~390 MB | TQ2_0_g128 (34B/128w) |
