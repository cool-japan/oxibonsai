# oxibonsai-runtime TODO

> Inference engine, sampling, tokenizer, OpenAI-compatible server
> Version 0.2.4 — 1,667 tests passing (all-features, 2026-07-22)

## Status: ✅ All Features Complete (Stable)

## 0.2.3 — Real Penalties/Logprobs, OpenAI-Compat Hardening, Two-Engine Speculative Decoding

- [x] **Real repetition/frequency/presence penalties** — `PenaltyParams` + `apply_frequency_presence_penalty` (OpenAI-style formula, range `[-2.0, 2.0]`); `Sampler::sample_with_history` applies the penalties over recent-token history; wired into `/v1/chat/completions`, the extended chat endpoint, and `/v1/completions` (previously `repetition_penalty` was a silent no-op and `frequency_penalty`/`presence_penalty` were honestly-400-rejected for lack of a seam) (`sampling.rs`, `server.rs`, `api_extensions.rs`, `completions.rs`)
- [x] **Real per-token `logprobs`/`top_logprobs`** — `InferenceEngine::generate_with_logprobs` captures per-step logits; wired into the extended chat endpoint, replacing the previous validated-but-null placeholder (`engine.rs`, `api_extensions.rs`)
- [x] **`/v1/completions` hardening** — honors `temperature`/`top_p`/penalties, reports the real loaded-model id (was hardcoded), batch `"prompt": [...]` now generates all prompts instead of effectively only the first, capped at `MAX_COMPLETION_BATCH_SIZE = 8` (`completions.rs`)
- [x] **Streaming/server-surface hardening** — mid-stream SSE failures emit a terminal `{"error":...}` event instead of a bogus clean finish (`stream_terminal_json`, `streaming.rs`); base `/v1/chat/completions` now parses `tool_calls` from generated text via `parse_base_tool_calls` (previously extended-endpoint-only); `<|...|>` ChatML markers in message content are stripped by default (`neutralize_special_markers`, opt-out `OXI_DISABLE_PROMPT_SANITIZATION`) to close a prompt-injection/turn-boundary-forgery gap; unified OpenAI-style `{"error":{...}}` envelope on every route (`http_error.rs`); real peer `ConnectInfo` + `RateLimitConfig::trusted_proxies` allowlist closes an `X-Forwarded-For`/`X-Real-IP` spoofing bypass in per-peer rate limiting (`server.rs`, `middleware.rs`, `rate_limiter.rs`)
- [x] **Grammar / JSON-Schema engine fixes** — previously-silently-ignored keywords (`minimum`/`maximum`/`minLength`/`maxLength`/etc.) now reject with `JsonSchemaCompileError::UnsupportedKeyword` instead of being ignored; real `const` support; optional (non-required) object properties are represented in the compiled grammar; `select_tool` now validates call arguments against the tool's schema via `validate_tool_arguments` (`grammar/json_schema_compiler.rs`, `tool_calling.rs`)
- [x] **Speculative decoding: real two-engine production path** — `SpeculativeDecoder::generate_verified` drafts against the draft engine's own KV and verifies against a **separate target `InferenceEngine`** (`verify_batch` / `forward_prefill_verify`), losslessly matching plain greedy decoding of the target; the old mock `generate_speculative` / harness `verify` are now `#[doc(hidden)]` / test-only primitives — no longer described as production APIs (`speculative.rs`)
- [x] **Prefix-cache / semantic-cache fixes** — `PrefixCachedEngine` keeps a persistent seeded sampler instead of discarding RNG state on every call; `SemanticCache::refit_embedder` re-embeds existing entries; `max_entries == 0` is clamped instead of silently disabling eviction; mutex sites recover from poisoning instead of panicking (`prefix_cache_engine.rs`, `semantic_cache.rs`)

**Known limitation (unchanged):** the OpenAI-compatible server always assembles a hardcoded ChatML prompt (`build_prompt` in `server.rs`) regardless of the loaded model. `oxibonsai-tokenizer`'s real multi-family `ChatTemplateKind` (ChatML/Llama-3/Mistral/Gemma/Qwen) exists but the server never calls it — fine for the Qwen3-based models this project ships today, but a documented gap for non-ChatML fine-tunes.

## Phase 15 — Extended Constraints + Grammar Engine

- [x] **`AllowListConstraint`** — finite-set token-sequence constraint (multiple-choice forcing); candidate bitmask + prefix tracking; `active_count()` accessor
- [x] **`SequenceConstraint`** — exact sequence forcing; `is_failed()` accessor; unconstrained after sequence consumed
- [x] **`LengthConstraint`** — hard `[min_len, max_len]` with optional stop token enforcement; `count()` accessor
- [x] **BNF grammar engine** (`src/grammar/`): `ast.rs` (Grammar AST, terminal normalisation), `bnf_parser.rs` (hand-rolled two-phase parser), `earley.rs` (full Earley recognizer with FIRST sets, `next_byte_set()`, `clone_state()`), `constraint.rs` (`GrammarConstraint` implementing `TokenConstraint`), `examples.rs` (5 pre-canned grammars)
- [x] **93 grammar tests** + **32 constraint tests** in `tests/`
- [x] **`AllowedTokensCache`** — LRU memoization cache for Earley `allowed_tokens` (Phase 15.x); `grammar/cache.rs`; `state_hash()` on `EarleyRecognizer`; `Mutex<AllowedTokensCache>` in `GrammarConstraint`; configurable capacity via `with_cache_capacity`; `cache_stats()` for observability; 12 tests in `tests/grammar_cache_tests.rs`

## Phase 19 — GBNF Parser + Tool Calling API

- [x] **GBNF parser** — `grammar/gbnf_parser.rs`: `parse_gbnf(src) -> Result<Grammar, GbnfParseError>`; two-pass (NT allocation then rule fill); expands `*`/`+`/`?` quantifiers to synthetic NTs; handles string literals, char classes `[...]`/`[^...]`, alternation `|`, grouping `(...)`; `GbnfParseError::{EmptyGrammar, MissingRootRule, UndefinedNonTerminal, DuplicateRule, UnexpectedChar, InvalidEscape, UnclosedGroup}`; re-exported from `grammar::` and `lib.rs`; 26 tests in `tests/gbnf_parse_tests.rs`
- [x] **Tool calling API** — `tool_calling.rs` (server-feature-gated): `select_tool(output, tools)` (XML tag parser + name registry check + JSON arg validation); `build_tool_constraint(tools)` (compile each tool schema → Grammar, merge with per-tool root rules); `make_tool_call(id, name, args)` convenience constructor; `new_tool_call_id()` alias; `ToolRegistry<'a>` for O(1) tool lookup; `validate_tool_arguments(args, tool)` structural required-field validator; `ToolCallError::{NoToolCallFound, UnknownTool, MalformedArguments, GrammarCompileError, EmptyToolList}`; 33 tests in `tests/tool_calling_tests.rs` + inline tests

## Phase 18 — Regex → BNF Compiler

- [x] **Regex → BNF compiler** — `grammar/regex_compiler.rs` (1,233 lines): `compile_regex(pattern) -> Result<Grammar, RegexCompileError>`; `ByteSet` (256-bit bitset, 4×u64); `RegexParser` (hand-rolled recursive descent); Thompson NFA construction; Subset DFA via powerset construction (2,048-state limit); DFA states → Grammar NTs with ε-productions for accepting states; supports literals, `.`, `[...]`/`[^...]` classes, `*`/`+`/`?`, `{n,m}`, `|`, grouping, `\d\w\s` escapes, `^$` anchors (ignored); `RegexCompileError::{InvalidSyntax, UnsupportedFeature, DepthExceeded, EmptyPattern, InvalidUtf8}`; re-exported from `grammar::` and `lib.rs`; 38 tests in `tests/regex_compile_tests.rs`

## Phase 17 — JSON Schema BNF + KV Cache Level

- [x] **JSON Schema → BNF compiler** — `grammar/json_schema_compiler.rs`: two-pass compiler; supports object/array/primitives/enum/anyOf/oneOf/allOf/$ref; `JsonSchemaCompileError`; re-exported from `grammar::` and `lib.rs`; 30 integration tests in `tests/json_schema_compile_tests.rs`
- [x] **`KvCacheLevel::Fp8`** — new variant (ordinal 2, between Q8 and Q4, `memory_factor=0.5`, `tag="fp8"`); all 4 match arms extended; 3 tests in `tests/kv_cache_policy_fp8_tests.rs`; existing metrics test updated to expect ordinal 3 for Q4

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

- [x] `InferenceEngine` — prefill + autoregressive decode loop
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
