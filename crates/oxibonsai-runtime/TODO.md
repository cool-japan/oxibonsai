# oxibonsai-runtime TODO

> Inference engine, sampling, tokenizer, OpenAI-compatible server
> Version 0.1.4 — 1,090+ tests passing (all-features, 2026-05-03)

## Status: ✅ All Features Complete (Stable)

## 0.1.4 — New Modules

- [x] **`kv_cache_policy`** — `KvCachePolicy` runtime controller; FP16/Q8/Q4 tier transitions driven by EWMA pressure with hysteresis (`kv_cache_policy.rs`, 14 tests)
- [x] **`adaptive_lookahead`** — speculative-decoding draft-length controller with cooldown + clamped `[min,max]` window (`adaptive_lookahead.rs`, 16 tests); wired into `SpeculativeDecoder::with_adaptive(...)`
- [x] **`request_metrics`** — per-request `RequestRateTracker` (TBT p50/p95, EWMA tok/s, queue-wait) plus `RequestRateAggregator` workload rollup (`request_metrics.rs`, 13 tests)
- [x] **`request_id`** — UUIDv4-style 128-bit identifier with thread-safe SplitMix64 generator (`request_id.rs`, 11 tests)
- [x] **Prometheus surface** — `oxibonsai_request_tokens_per_second`, `oxibonsai_inter_token_latency_p{50,95}_seconds`, `oxibonsai_queue_wait_seconds`, `oxibonsai_kv_cache_compression_level` gauges added to `InferenceMetrics`

## 0.1.4 — Engine Integration

- [x] **`InferenceEngine::generate_tracked(...)`** — populates a `RequestRateTracker` during generation, pushes the snapshot to an attached `RequestRateAggregator`
- [x] **`InferenceEngine::generate_with_request_id(...)`** — emits a tracing span tagged `request_id = <uuid>`, returns `(Vec<u32>, RequestRateTracker)` for client-side telemetry
- [x] **`InferenceEngine::set_rate_aggregator(Arc<RequestRateAggregator>)`** — workload-aggregator setter
- [x] **`tests/engine_controllers_tests.rs`** — 8 integration tests covering tracked generate, request-id propagation, aggregator push semantics
- [x] **`examples/runtime_controllers.rs`** — end-to-end demo (in workspace root)
- [x] **`benches/controllers_bench.rs`** — criterion microbenchmarks (in workspace root)

## 0.1.4 — Server / Admin Integration

- [x] **`GET /admin/workload-stats`** — combines `RequestRateAggregator` snapshot (TBT p50/p95, EWMA tokens/sec, queue-wait, completed requests) with `KvCachePolicy` state (level, pressure, upgrades, downgrades) into a single operator-friendly JSON document
- [x] **`AdminState::with_rate_aggregator(Arc<RequestRateAggregator>)`** + **`AdminState::with_kv_cache_policy(Arc<KvCachePolicy>)`** — builder-style attachment of workload sources to admin
- [x] **`RequestId::as_bytes() / from_bytes()`** — 16-byte big-endian round-trip for binary protocols
- [x] **`X-Request-ID` HTTP header propagation** in the OpenAI server: client-supplied ids are echoed; missing/malformed → auto-generated. Public helpers `resolve_request_id` / `request_id_header_map` and `REQUEST_ID_HEADER` constant. Streaming + non-streaming both carry the header; server tracing spans now record `request_id` for end-to-end correlation. 8 integration tests in `tests/request_id_propagation_tests.rs`

Observability, TOML config, streaming SSE, circuit breaker, health checks, builders, presets, batch engine, async engine, continuous batching, prefix/semantic caches, speculative decoding, beam search, token healing, advanced/adaptive sampling, quality metrics, memory profiling, RAG server, and WASM support all implemented.

## Done

- [x] `Engine` / `InferenceEngine` — prefill + autoregressive decode loop
- [x] `InferenceEngine::from_gguf()` — load model from GGUF file
- [x] `Sampler` — temperature, top-k, top-p, repetition penalty, `LcgRng`
- [x] `TokenizerBridge` — HuggingFace tokenizers wrapper (encode/decode)
- [x] Native tokenizer — in-tree BPE/SentencePiece decoding
- [x] OpenAI-compatible `/v1/chat/completions` endpoint (non-streaming + streaming)
- [x] `/v1/completions`, `/v1/embeddings`, `/v1/models`, `/health` endpoints
- [x] RAG endpoints (`/v1/rag/*`) and admin API (`/admin/*`)
- [x] CLI subcommands: `run`, `chat`, `serve`, `info`
- [x] **Tracing upgrade** — `EnvFilter` + optional JSON layer (`tracing_setup.rs`)
- [x] **`#[instrument]` spans** — `generate()`, server handlers; span hierarchy: request → prefill → decode
- [x] **Prometheus metrics** — `/metrics` endpoint; tokens generated, requests, tokens/sec, prefill latency, decode latency, request latency (`metrics.rs`)
- [x] **TOML config struct** — Server settings, sampling defaults, model path, tokenizer path, observability settings (`config.rs`)
- [x] **Layered config** — defaults → TOML file → CLI args override (`config.rs`)
- [x] **Streaming chat completions** — SSE via `tokio-stream` (`server.rs`, `streaming.rs`, `stream_metrics.rs`)
- [x] **Circuit breaker** — Fault isolation for engine errors (`circuit_breaker.rs`)
- [x] **Rate limiter & middleware** — token-bucket limiter and tower middleware (`rate_limiter.rs`, `middleware.rs`)
- [x] **Health checks** — Liveness and readiness probes (`health.rs`)
- [x] **Builders** — Ergonomic `EngineBuilder` and server builder (`builders.rs`)
- [x] **Presets** — Greedy / Balanced / Creative / Code sampling presets (`presets.rs`)
- [x] **Batch engine** — Batch inference for throughput optimization (`batch_engine.rs`)
- [x] **Continuous batching** — streaming batch scheduler (`continuous_batch.rs`, `request_queue.rs`)
- [x] **Async engine** — Non-blocking async inference paths (`async_engine.rs`)
- [x] **Recovery** — Error recovery and retry strategies (`recovery.rs`)
- [x] **Convenience helpers** — High-level one-shot inference API (`convenience.rs`)
- [x] **InferencePipeline** — stop reasons, streaming, token budget (`pipeline.rs`, `token_budget.rs`)
- [x] **Advanced samplers** — Mirostat v1/v2, Min-P, Eta, Locally Typical, SamplerChain (`sampling_advanced.rs`)
- [x] **Adaptive sampling** — runtime-tuned sampling (`adaptive_sampling.rs`)
- [x] **Speculative decoding** — draft/verify loop (`speculative.rs`)
- [x] **Beam search** — configurable width, length penalty, n-gram blocking (`beam_search.rs`, `ngram_cache.rs`)
- [x] **Token healing & constrained decoding** — JSON schema guided output (`token_healing.rs`, `constrained_decoding.rs`, `json_schema.rs`)
- [x] **Context manager** — sliding window and KV reuse (`context_manager.rs`)
- [x] **Prefix cache engine** — reusable KV prefixes (`prefix_cache_engine.rs`)
- [x] **Semantic cache** — embedding-based response cache (`semantic_cache.rs`, `embedding_index.rs`)
- [x] **Model cache & multi-model** — hot-swap and concurrent models (`model_cache.rs`, `multi_model.rs`, `hot_reload.rs`)
- [x] **Auto-tuner & quality metrics** — runtime tuning and eval metrics (`auto_tuner.rs`, `quality_metrics.rs`)
- [x] **Memory profiler** — RSS via Mach (macOS) / statm (Linux) (`memory.rs`, `profiler.rs`)
- [x] **Deduplication & n-best** — request dedup and beam n-best output (`dedup.rs`, `nbest.rs`)
- [x] **Distributed runtime** — sharded inference primitives (`distributed.rs`)
- [x] **WASM API** — browser-safe subset behind `wasm` feature (`wasm_api.rs`)
- [x] **Web UI** — lightweight embedded console (`web_ui.rs`)
- [x] **Integration tests** — `tests/generate_pipeline_tests.rs`: full generate() pipeline, determinism, sampling params, edge cases, engine state
- [x] **Sampling distribution tests** — `tests/sampling_distribution_tests.rs`: chi-square goodness of fit, temperature/top-k/top-p/repetition penalty statistical validation
- [x] **Feature matrix** — `server`, `rag`, `wasm`, `metal`, `native-cuda` all green under all-features (2026-04-18)
