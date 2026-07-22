# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.3] - 2026-07-21

Large cross-cutting hardening pass following a 109-agent production-release audit (91 confirmed findings across 23 implementation packages plus documentation honesty). Full per-package evidence lives in `TODO.md`'s "Production-Release Audit (2026-07-20)" section.

A second pass (2026-07-21, same branch) closed the remaining Metal GPU platform gaps (Q4_0/Q8_0/K-quant GEMV + model-layer dispatch, FP8 batch prefill), built and wired the OpenAI-compatible penalty/logprobs seam Round 1 had only validated-and-honestly-rejected, added NEON Q4_0/Q8_0 kernels, fixed a kernel-dispatch-threshold gap, and closed further GGUF/checkpoint/LoRA/tokenizer/server robustness and documentation-accuracy gaps. Full per-package evidence lives in `TODO.md`'s "Production-Release Audit Round 2 (2026-07-21)" section; every superseded Round-1 claim is annotated in place there rather than silently rewritten.

### Fixed

- **CUDA prefill/decode KV-cache desynchronization (P0)** — unified the batch-prefill KV cache with the per-token decode KV cache (`cuda_prefill/state.rs` now delegates to `cuda_full_layer::acquire_kv_cache`), fixing silent decode corruption on Q1 prompts ≥17 tokens (decode logit Δ collapsed from ~7.3 to ~0.001); fixed the missing per-token `d_pos_seqlen` upload in the Q1 prefill loop; re-enabled ternary CUDA batch prefill, which had been disabled since the same bug was found.
- **`InferencePipeline` generation** now honors `GenerationStrategy`, applies the attached `TokenConstraint`, wires token healing against the real model, and no longer panics on engine errors (previously used `.expect()`).
- **Speculative decoding** gained a real, non-fabricating two-engine entry point, `SpeculativeDecoder::generate_verified`, that drafts against the delta-KV path and verifies against a genuine separate target `InferenceEngine`; fixed a draft-reuse out-of-bounds panic.
- **Tokenizer fidelity**: special/added tokens are now protected from BPE/Unigram/WordPiece shredding at `encode()` dispatch; GPT-2 byte-level bytes↔unicode mapping is wired in (fixes CJK/emoji/accented-Latin encoding and whitespace-run preservation); BPE merge-rank lookup is now O(1) instead of an O(total-merges) scan.
- **`JsonConstraint`/`RegexConstraint`** now operate on real token text (via a new decoder-backed `TokenTextIndex`) instead of misinterpreting raw token IDs as Unicode codepoints.
- **Server request validation**: `max_tokens`/temperature/`top_p`/`n` are now validated with honest 400s (closes an unbounded-`max_tokens` OOM/DoS path); `/rag/query` performs a real encode→generate→decode instead of returning a fabricated single token; `/v1/models` and `/admin/config`/`/admin/cache-stats` now report real, honest state instead of hardcoded/fake values; the admin router, chat web UI, CORS, request logging, and an opt-in rate limiter are now actually mounted on the served router.
- **OpenAI-compatible API extensions**: `finish_reason` is now derived rather than hardcoded; `frequency_penalty`/`presence_penalty` are now rejected with an honest 400 instead of being silently discarded (there is no penalty seam in the sampler to apply them to); `logprobs` on the extended chat endpoint now honestly returns `null` instead of a fabricated empty array; `n>4` now 400s.
- **GGUF parsing hardening**: bounded Array-nesting recursion depth, checked/saturating tensor offset/size/shape arithmetic, and `general.alignment` validation against untrusted/adversarial GGUF files; the existing compat-report checker is now wired into the real load path.
- **Quantization/export correctness**: fixed a Q1_0_g128 export sign-bit inversion (a real bit-level correctness bug, not just a doc issue); `Int8PerChannel` export now refuses instead of silently mislabeling its bytes as `F32`; guarded a division-by-zero in per-channel quantization; bounded `Checkpoint::read_from`'s eager allocations against corrupted headers; extended f32 tensor loading to cover Q4_K/FP8 E4M3/E5M2.
- **`oxibonsai-serve`**: `max_concurrent_requests`/`per_request_timeout_ms` are now genuinely enforced (tower admission stack: overload→503, timeout→408); `tokenizer.kind` is validated against a real backend whitelist; CLI-flag precedence over TOML/env is now correct even when the CLI value matches the built-in default; bearer-token comparison is now constant-time.
- **Grammar/JSON-Schema engine**: the JSON-Schema→BNF compiler now rejects previously-silently-ignored keywords (`minimum`/`maximum`/`minLength`/`maxLength`/`minItems`/`maxItems`/`uniqueItems`) and implements real `const` support; object schemas now support optional (non-required) properties; tool-calling's `select_tool` now validates arguments against the tool's schema instead of only checking well-formed JSON; the Earley `allowed_tokens` cache is now keyed by `(state_hash, vocab_size)`, closing a latent wrong-length-mask risk.
- **Prefix-cache engine / semantic cache**: `PrefixCachedEngine` now keeps a persistent seeded sampler across calls instead of discarding RNG state every call; `SemanticCache::refit_embedder` re-embeds existing entries instead of leaving stale-dimension zombies; `max_entries == 0` no longer produces undefined eviction behavior; all server-reachable mutex sites recover from poisoning instead of propagating a panic.
- **`oxibonsai quantize`** now performs a real dequantize→re-encode→write pipeline instead of fabricating size/ratio estimates from the format string alone.
- **TruthfulQA / multiple-choice evaluators**: empty or all-NaN logits no longer silently default to "index 0 is correct"; `ChunkConfig` now rejects `min_chunk_size > chunk_size`, which previously guaranteed permanently-empty chunking on every RAG indexing path.
- **`KvCache::store_key`/`store_value`** gained real release-mode bounds checking (previously `debug_assert!`-only, a no-op in release builds); `PagedKvCache::sequence_length()` now returns the actual highest-written position instead of block-rounded capacity.
- **`wasm32-unknown-unknown` build restored** — `engine_pool` (which unconditionally used `tokio`) is now correctly cfg-gated off the wasm target, matching the sibling `async_engine` module.
- **`oxibonsai` facade crate**: the `server`/`full` Cargo features now actually compile (previously missing `dep:oxibonsai-serve`).
- **`kernels-cpu`**: guarded `AlignedBuffer`/`AlignedBlocks` allocation sizing against `usize` overflow; removed dead AVX-512 gather/scatter helpers and wired a previously-dead AVX-512 streaming-store GEMV kernel into the dispatch tier; `PlatformProfile`'s tuned thresholds now genuinely govern GEMV/GEMM parallel dispatch.
- **`kernels-meta`**: `build.rs`'s embedded-metallib fast path pointed at a deleted source file and was permanently dead; it now reads the current `kernel_sources/` module layout, with a loud panic on future kernel-whitelist drift instead of a silently empty metallib.
- **Image pipeline**: fixed a `TileBoundary::AfterMid` no-op and added TE GPU weight residency on CUDA.

**2026-07-21 second pass:**

- **`repetition_penalty` was a silent no-op** (`oxibonsai-runtime`) — the default-enabled `SamplingParams::repetition_penalty` (1.1) is now genuinely applied in the native decode loop via a new `Sampler::sample_with_history` + `PenaltyParams` (`frequency_penalty`/`presence_penalty`, OpenAI formula) seam; the no-penalty case remains bit-identical to prior output. Also added GGUF-resolved `eos_token_id()` (was always the hardcoded fallback) and a logits-capturing `generate_with_logprobs`.
- **OpenAI-compatible penalties/logprobs are now genuinely applied**, not honestly rejected: base `/v1/chat/completions` and the extended endpoint both apply `frequency_penalty`/`presence_penalty` (`[-2.0, 2.0]`) for real (Round 1's 400-rejection is retired) and return real per-token `logprobs`/`top_logprobs` (was a validated-but-null placeholder on the extended endpoint); `/v1/completions` now also honors `temperature`/`top_p`/penalties and reports the real loaded-model id (was hardcoded `"bonsai-8b"`). A mid-stream SSE failure now emits a terminal `{"error":...}` event instead of a bogus clean finish; the base endpoint now parses `tool_calls` from generated text (previously extended-only); `<|...|>` ChatML markers in message content are now stripped by default before prompt assembly (prompt-injection/turn-boundary-forgery fix, opt-out `OXI_DISABLE_PROMPT_SANITIZATION`); every route now returns a unified OpenAI-style `{"error":{...}}` envelope; the server now serves with real peer connect-info so per-peer rate limiting is live end-to-end.
- **Metal GPU platform gaps closed**: new Metal GEMV kernels + model-layer dispatch for Q4_0/Q8_0/K-quant (Q2_K–Q8_K) — `Linear*::forward()` now tries Metal first (mirroring the existing CUDA short-circuit) before falling back to CPU; a hybrid GPU-linear-projections/CPU-attention Metal FP8 batch prefill (Phase 28.B) that avoids the CUDA-FP8-style split-KV-cache bug by construction. Both parity-validated on Apple-Silicon M3.
- **NEON GEMV kernels for Q4_0/Q8_0** (`oxibonsai-kernels`) implemented and wired into the runtime dispatcher — was silently falling back to the scalar reference on AArch64 CPUs.
- **Kernel-dispatch thresholds**: the adaptive tile-size selector (`parallel_tiled.rs::select_gemv_strategy`) and L1 tile sizing (`tiled.rs::optimal_tile_rows`) now read `PlatformProfile::global_thresholds()` instead of hardcoded constants — closes a gap in the earlier "thresholds now govern dispatch" claim, which had only covered `parallel.rs`.
- **Metal decode robustness**: context-length guards on all six Metal prefill/greedy entry points (was an out-of-bounds RoPE-table panic on an over-long prompt); a coherent Metal→CPU mid-stream fallback in `generate_greedy_gpu` that rebuilds the CPU KV cache before continuing (was silent corruption reading a stale/zero CPU cache); `gpu_cache.rs`'s Q1 GPU-handle fallbacks are now hard errors with a CPU fallback instead of silently defaulting to handle `0`; `CachedModelWeights` split into a `Q1`/`Ternary` enum, removing dead cross-type fields.
- **GGUF parser hardening follow-ups**: metadata/tensor-array eager allocation is now capped independent of the attacker-declared count in both the batch and streaming parsers; string parsing uses a bounded 64 KiB-chunk incremental read instead of an upfront allocation; duplicate metadata keys/tensor names are now a hard parse error in both parsers (was silent last-wins); `GgufWriter::write` now validates alignment and shape-product/offset arithmetic instead of silently wrapping/overflowing.
- **`Checkpoint::read_from`** now cross-validates each tensor's declared shape against its actual data length (new `CheckpointError::ShapeDataMismatch`); **`SmoothQuantCalibrator::record_activation`** returns `Result` instead of panicking on an `in_features` mismatch; **`LoraAdapter::apply`/`merge_into_weights`** return `Result` instead of silently truncating or panicking on a dimension mismatch (breaking API changes; LoRA adapters are not wired into the live inference forward path, so there is no runtime behavior impact).
- **Tokenizer fidelity follow-ups**: non-special `added_tokens` (e.g. Qwen's `<tool_call>`, `<|fim_prefix|>`/`<|fim_middle|>`/`<|fim_suffix|>`) now encode atomically instead of being shredded by BPE (closes a gap in the earlier special-token-protection fix, which only covered `special: true` tokens); HF `Sequence`-wrapped byte-level pre-tokenizer/decoder detection now recurses correctly (previously silently mis-detected on affected configs, corrupting CJK/emoji/accented-Latin decode output); the GPT-2 unicode↔byte map is now cached once per process instead of rebuilt per decode call; BPE vocabulary construction now rejects two token strings claiming the same numeric ID instead of a non-deterministic `HashMap`-order decode winner.
- **`oxibonsai serve` (root CLI binary)** now mounts the same bearer-auth (constant-time) + admission-control (`--max-concurrent-requests`, `--request-timeout-ms`) hardening the standalone `oxibonsai-serve` binary already had.
- **`/v1/embeddings`'s `encoding_format: "base64"`** now emits real RFC 4648 base64 (was silently-wrong hex mislabeled as base64).
- **`/rag/query`'s `top_k`** now genuinely drives both retrieval and the generation prompt (was a dead-code override always using the pipeline's fixed default); **`/rag/index`'s `chunk_overlap`** must now satisfy `<= chunk_size / 2` (previously unbounded, up to ~500× worst-case storage amplification).
- **Rate-limit peer-IP spoofing**: new `RateLimitConfig::trusted_proxies` allowlist closes the spoofable `X-Forwarded-For`/`X-Real-IP` rate-limit bypass — those headers are now only honored from an allowlisted reverse-proxy peer.
- **`/v1/completions` batch prompts** (`"prompt": [...]`) now all generate (previously effectively only the first), one choice per prompt, capped at 8 prompts per request.
- **`oxibonsai convert --quant q1_0_g128`** is now genuinely implemented end-to-end in both the HF-safetensors and ONNX converters (previously `tq2_0_g128`-only despite `--help` briefly claiming otherwise mid-audit).
- **CLI model-open errors** (`run`/`chat`/`info`/`quantize`/`validate`) now include the offending path in the error message.
- **`AlignedBuffer`/`AlignedBlocks`** (`oxibonsai-kernels::packing`) guarded against `usize` allocation-size overflow, mirroring the sibling `aligned` module's existing guard; removed a dead duplicate `prefetch_read`/`prefetch_write` implementation.

### Added

- **Q4_0/Q8_0 tiered CPU kernels** (AVX2 + AVX-512 GEMV, plus Rayon row-parallelism) and **Rayon row-parallel K-quant (Q2_K–Q8_K) GEMV**.
- **A real GitHub Actions CI pipeline** (`.github/workflows/ci.yml`): fmt, clippy `-D warnings`, nextest, a wasm32-unknown-unknown compile-check, a macOS `--features metal` compile-check, a native-cuda compile-check, and a manually-triggered self-hosted-GPU test job.
- **`oxibonsai serve --rag`** mounts the RAG HTTP API (`/rag/index`, `/rag/query`, `/rag/stats`) on the CLI's own server, previously only reachable through a separate, unbuilt code path.
- Property/roundtrip test coverage for Q3_K/Q5_K/Q6_K/Q8_K/TQ2_0(_g128)/FP8 E4M3+E5M2/Q4_0/Q8_0 (`oxibonsai-core`), and an always-on CPU-vs-CUDA engine-level determinism regression test spanning the 16-token batch-prefill boundary.
- `oxibonsai-cli` crates.io metadata (`homepage`, `documentation`, `keywords`, `categories`).
- **`oxibonsai eval` CLI subcommand** (Cargo feature `eval`, 2026-07-21): scores a model's generations against a JSONL reference dataset (`{"input":...,"expected_output":...}`) with ROUGE-1/2/L, with optional `--report-json`/`--report-markdown` output.
- **`deny.toml`** (workspace root, 2026-07-21): enforces the COOLJAPAN banned-crate list (`openblas`/`bincode`/`rustfft`/`z3`/`rusqlite`/`zip`/`flate2`/`zstd`/`bzip2`/`lz4`/`tar`/`snap`/`brotli`/`miniz_oxide` and `-sys`/`-src` variants) plus an OSS license allowlist; `cargo deny check bans licenses sources` is now a gate in both `scripts/publish.sh` and CI.
- **`docs/DEPLOYMENT.md`** (new, 2026-07-21): production deployment guidance — binary choice, reverse-proxy/TLS termination, bearer auth, admission limits, Prometheus scraping.
- **NEON GEMV kernels for Q4_0/Q8_0** (`oxibonsai-kernels::simd_q_std_neon`, 2026-07-21), wired into `KernelDispatcher`.
- **`PenaltyParams`, `apply_frequency_presence_penalty`, `Sampler::sample_with_history`** (`oxibonsai-runtime`, 2026-07-21) — public frequency/presence-penalty seam.
- **`InferenceEngine::generate_with_logprobs`** (`oxibonsai-runtime`, server-gated, 2026-07-21) — logits-capturing generation returning real per-token `LogprobsContent`.

### Changed

- **Q4_0/Q8_0 and K-quant (Q2_K–Q8_K) CUDA batch prefill are now disabled by default** (clean `Err` → sequential fallback) after an audit found the same prefill/decode split-KV-cache bug that was fixed for Q1/ternary above is also present in these paths; re-enablement is pending the same KV-cache unification fix. Functional correctness is preserved (sequential fallback is bit-correct); only batch-prefill throughput for these formats is affected.
- The original `SpeculativeDecoder::generate_speculative`/`verify` mock two-engine harness (synthetic draft probability, synthesized target logits) is now `#[doc(hidden)]` and documented purely as a test harness; it is no longer presented as a production speculative-decoding path.
- `scripts/publish.sh`'s crate publish order now includes `oxibonsai-image` in a verified topological position.
- **`--guidance`/`:guidance` removed** (2026-07-21) from `oxibonsai image`/`oxibonsai repl` and the REPL command set — Bonsai-Image is a distilled, CFG-free model (`guidance_embeds: false`), so the flag was provably inert.
- **`oxibonsai convert`'s `general.quantization_version` GGUF metadata** (2026-07-21) now records the actually-written quant format instead of always being hard-coded to `"TQ2_0_G128"`.

### Documentation

- **README.md**: added a "Known Limitations" section consolidating every claims-vs-reality gap found by this audit (scirs2 CUDA tier is a CPU fallback; multi-GPU is a rayon simulation; distributed serving has no networking; PagedAttention / KV-cache quantization / attention-variant layers — sink, sparse, flash-decode, cross-attention, MoD, MoE — / RoPE-scaling variants are tested standalone primitives not wired into the default forward path; Metal FP8 batch-forward integration and Metal Q4_0/Q8_0/K-quant kernels remain platform gaps pending macOS hardware; the live, unresolved FP8-CUDA batch-prefill KV-cache bug found during this pass). Annotated the Acceleration Tiers table and Development Roadmap table accordingly.
- **TODO.md**: flipped all 23 audit-package checkboxes to done with outcome summaries; annotated every Phase-status line whose claim no longer matches the shipping forward path (multi-GPU, distributed serving, PagedAttention, KV-cache quantization, attention-sink/sparse-attention/flash-decode/cross-attention/MoD/MoE, RoPE-scaling/YaRN, `KvCachePolicy`, OpenAI penalty/logprobs claims, and the CUDA Q4_0/Q8_0/K-quant/FP8 batch-prefill regressions); recorded the `gpu_cache.rs` enum-split hygiene plan (zero functional impact today, macOS-only validation); added two new open items surfaced by this pass (a live FP8-CUDA batch-prefill KV-cache bug, and an FP8 E5M2 quantization block-scale rounding defect that can silently produce `Infinity`).
- **docs/CLI.md**: corrected the `quantize` command's stale "simulates the operation" description to match its real dequantize/re-encode/write behavior, and updated its supported-format list; documented the `serve --rag` flag.
- **2026-07-21 second pass — README.md**: Known Limitations updated to remove the now-closed Metal FP8/Q4_0/Q8_0/K-quant platform gaps, and to add two newly-disclosed standalone/unwired items: sliding-window attention (`TransformerBlock::forward_with_sliding_window`) and the OpenAI-compatible server's hardcoded-ChatML prompt template (no per-model `chat_template` support, despite `oxibonsai-tokenizer` having a real multi-family `ChatTemplateKind` the server never calls).
- **2026-07-21 second pass — TODO.md**: new "Production-Release Audit Round 2 (2026-07-21)" section covering all 3 waves of this pass; every Round-1 claim this pass superseded is annotated in place (the `bonsai-8b` stale-test tracking item, `RagState` pooling, the `gpu_cache.rs` enum-split plan and handle-fallback item, the Metal Q4_0/Q8_0/K-quant/FP8 platform-gap items, the `kernels-cpu-02` "thresholds now govern dispatch" overclaim, and the OpenAI penalty/logprobs 400-rejection claims); recorded the `.github/workflows/ci.yml` policy-exception note; added the newly-surfaced sliding-window-attention disclosure gap.
- **2026-07-21 second pass — docs/CLI.md**: removed the retired `--guidance`/`:guidance` flag from the `image`/`repl` tables and the REPL command table; documented `oxibonsai serve`'s new `--bearer-token`/`--max-concurrent-requests`/`--request-timeout-ms` hardening flags and the new `oxibonsai eval` subcommand.

---

## [0.2.2] - 2026-06-08

### Added
- **`oxibonsai repl` interactive image REPL** (`oxibonsai-image`, `oxibonsai-cli`): `ImageSession`
  loads the DiT, VAE, and text encoder **once** and renders many prompts without re-paying the
  load/dequant cost. `StageTimings` and `RenderOutcome` surface per-stage wall-clock splits.
  The session puts the text encoder in resident mode (`TeWeights::set_resident`) so the
  dequantised f32 weights (~16 GB) stay cached across renders on high-memory machines.
  On Ghostty the rendered image is shown inline via the kitty graphics protocol; on other
  terminals the PNG is written to a file. Runtime commands: `:steps`, `:seed`, `:size`,
  `:fast` (2-step 384×384 preview), `:hq` (8-step 512×512), `:out`, `:open`, `:help`, `:quit`.
- **`TeWeights::set_resident(on: bool)`** (`oxibonsai-image`): controls whether the Mlx4bit
  source caches dequantised f32 tensors across forwards. Off by default (preserves the one-shot
  CLI low-RAM profile); turned on by `ImageSession` for the REPL use-case.
- **Kitty graphics protocol support** (`src/cli/term.rs`): pure-Rust base64 encoder and inline
  PNG display for Ghostty terminals (`kitty_supported()` auto-detects via `GHOSTTY_*` env vars
  and `TERM`/`TERM_PROGRAM`).
- **GPU acceleration flags documented in `.env.example`**: `OXI_DIT_ATTN_GPU` (flash-attention,
  default-ON on Apple Silicon), `OXI_VAE_GPU` (convolutions, default-ON on Apple Silicon), and
  `OXI_TE_GPU` (text-encoder GPU, default-OFF — CPU SIMD wins on Apple Silicon; may help on
  Windows/NVIDIA CUDA) with platform-specific comments.
- **CUDA TQ2 GEMV parity test** (`oxibonsai-kernels`): isolated probe `cuda_tq2_gemv_parity.rs`
  for Blackwell GPU output validation; compile-gated behind `cfg(feature = "cuda")`.

### Changed
- **`decoded_chw_to_rgb8` extracted as shared helper** (`oxibonsai-image/pipeline.rs`): CHW→HWC
  f32-to-u8 conversion factored into a `pub(crate)` function, shared by both `text_to_image` and
  `ImageSession::render` to guarantee byte-identical pixel output from both paths.
- **`oxionnx-proto` bumped 0.1.3 → 0.1.4** (`Cargo.toml` workspace dependencies).

---

## [0.2.1] - 2026-06-06

### Changed

- **Raised compile optimization for the `test` and `dev` profiles** (`Cargo.toml`): added `[profile.test] opt-level = 2` and `[profile.dev.package."*"] opt-level = 3`, so test binaries and all dependencies (including workspace path-deps like `oxibonsai-model` / `oxibonsai-kernels` when built as deps of another crate's tests) are optimized and autovectorized. At the default `opt-level = 0` the workspace's tests run real numeric work unoptimized — the parity golden references and model forward passes took *minutes* (e.g. the DiT-shape joint-attention CPU reference ~14.5 GFLOP, and the speculative decoder's ~240 forward passes over the 151936-row vocab). The crate under active edit stays at `dev` opt-level 0, so incremental compiles of your own code remain fast. Float results are unchanged: Rust does not enable fast-math, so `opt-level` does not reassociate reductions — parity gates (cos≥0.999) stay bit-stable.
- **Bumped `oxiarc-deflate` 0.3.2 → 0.3.3** (`Cargo.toml`, `[workspace.dependencies]`): tracks the latest OxiARC release per the COOLJAPAN Latest-crates policy. `oxiarc-deflate` provides the Pure-Rust DEFLATE backend for PNG output in `oxibonsai-image`; the substantive fixes in the 0.3.3 OxiARC release land in sibling crates (`oxiarc-brotli` high-entropy round-trip), so for OxiBonsai this is a version-tracking bump with no change to DEFLATE/PNG behavior.

### Fixed

- **VAE precheck rejected a valid `.safetensors` file** (`oxibonsai-image`, `src/pipeline.rs`): the text-to-image precheck used `is_dir()`, so a valid `.safetensors` FILE passed via `--vae` / `OXI_VAE_WEIGHTS` was rejected with "VAE weights dir not found" before any loading — even though `VaeWeights::open` and the docs both accept a file. The precheck now accepts a file **or** a directory (`is_file() || is_dir()`), error wording is corrected, and the stale doc-comment is fixed. Added regression tests `test_issue_9_*`. (#9)

### Documentation

- **Corrected stale HuggingFace asset paths** (`docs/IMAGEN.md`, `crates/oxibonsai-image/README.md`): the DiT lives under `transformer-packed-mflux/`, and the text encoder + tokenizer ship inside the main `prism-ml/bonsai-image-ternary-4B-mlx-2bit` repo under `text_encoder-mlx-4bit/` (the standalone `prism-ml/text_encoder-mlx-4bit` repo does not exist). Consumer paths updated to match `hf download --local-dir` layout, and the bundled vs. gated `black-forest-labs/FLUX.2-dev` VAE choice is clarified. Verified against the live HF API. (#8)

---

## [0.2.0] - 2026-06-02

### Added

- **Concurrent engine pool for `/serve`** (`oxibonsai-runtime`, `src/engine_pool.rs`): `EnginePool` wrapping `Vec<InferenceEngine>` with `Arc<Semaphore>` permits. CPU tier = N replicas sharing one `Arc<[f32]>` token-embedding table (eliminates N duplicate ~1.16 GB allocations); GPU tier = clamped to 1 (Metal process-global singleton). `build_pool_from_gguf` API. Default CPU pool size: `min(4, num_cpus)`.

- **CPU-vs-Metal parity guard** (`oxibonsai-runtime`): `InferenceEngine` validates byte-identical greedy output across CPU and Metal backends at startup. Validated byte-identical on real 1.7B Ternary-Bonsai model.

- **CUDA imagen backend** (`oxibonsai-kernels`, `oxibonsai-image`): native CUDA GPU backend for imagen on Linux/Windows, authored as a blind mirror of the Metal path; parity-first plain-FP32, cap-of-8-safe, additive (Metal byte-unchanged). Three acceleration tiers: `gemm_tq2` kernel retile (32×32 → 128×128 transposed-shared, ~6× kernel speedup); flash-attention warp-cooperative (lane-split head_dim + `__shfl_xoc_sync`, ~6.3× kernel speedup); Stage-0 context_embedder GPU port (`encode_gemm_f32`, 59× on that operation). Steps=4 benchmark projection on A4000-class hardware: ~101s → ~31.7s (3.2×). Compile + cos≥0.999 parity validation deferred to Linux/CUDA hardware.

- **Text Encoder 4-bit** (`oxibonsai-image`): `open_mlx_4bit` loads native 2.1 GB MLX 4-bit safetensors directly (was 15 GB f32 `.npy`). Dequantizes `mlx-packed-affine` 4-bit weights to f32 on demand. Real Bonsai-Image footprint ≈3.5 GB. Parity gate: `te_parity` cos≥0.999999 vs MLX oracle. Activated via `OXI_TE_4BIT` env var.

- **`oxibonsai image` CLI subcommand with `--seed` reproducibility** (`oxibonsai-image`): `oxibonsai image --prompt … --seed N --out x.png` standalone text-to-image command. MLX-exact Threefry-2×32 RNG port (`src/sample/mlx_rng.rs`) — byte-exact vs official mflux reference with `--seed 42`. Env vars: `OXI_DIT_GGUF`, `OXI_VAE_WEIGHTS`, `OXI_TE_4BIT`, `OXI_TE_TOKENIZER_DIR`.

- **Stable-toolchain compatibility for `oxibonsai-kernels`**: previously required nightly Rust for `#[feature(stdarch_aarch64_prefetch)]`. Now builds cleanly on stable (1.86+) and nightly via `build.rs` nightly-detect setting `cfg(nightly_aarch64_prefetch)`, and `aarch64_prefetch!` macro (no-op on stable, active prefetch on nightly; zero new warnings either way).

### Fixed

- **Temperature-discard bug** (`oxibonsai-runtime`, `src/completions.rs`): temperature was being silently discarded from `SamplingParams` on the streaming completions path. Now threaded correctly through to sampling.

---

## [0.1.5] - 2026-06-02

### Added

- **New crate: `oxibonsai-image`** — complete Pure-Rust text-to-image pipeline for PrismML Bonsai-Image (FLUX.2-Klein 4B); first pure-Rust, C/C++/Fortran-free, zero-FFI implementation of the Bonsai-Image FLUX.2-Klein text-to-image pipeline, built entirely on the COOLJAPAN ecosystem.

- **DiT: `Flux2Transformer2DModel`** (`oxibonsai-image`): 5 double-stream + 20 single-stream blocks, TQ2_0_g128 ternary weights, 128-channel latents. 4-axis RoPE, AdaLN modulation, fused gate+up+SwiGLU, joint/single attention. Parity gate: `dit_parity` 59 taps cos≥0.999.

- **Ternary GEMM v10 kernel** (`oxibonsai-kernels`): f16-D-EXACT staging (code×scale∈{−s,0,+s}), vectorized dequant-scatter. ~1.89× over v9 end-to-end DiT on Apple Silicon Metal. All 59 dit_parity taps cos≥0.999.

- **Flash-attention Metal kernel** (`oxibonsai-image`, `oxibonsai-kernels`): `dit_attention_flash.rs` — simdgroup f32 MACs, flash-v2 online-softmax. 5.47× vs CPU rayon+NEON reference (59ms vs 323ms per step); parity cos=1.0 (59 taps cos≥0.999). Default-on for DiT inference.

- **`AutoencoderKLFlux2` VAE decoder** (`oxibonsai-image`): Pure-Rust Conv2d (im2col + GEMM), GroupNorm(32), SiLU, ResNet blocks (layers_per_block=2), 4-stage upsample (128/256/512/512), post_quant_conv, patch[2,2] unpack, fp32 (force_upcast). Parity gate: `vae_parity` 11 taps cos≥0.999.

- **Metal GPU VAE** (`oxibonsai-image`): default-on GPU VAE decode (`OXI_VAE_GPU`, opt-out env="0"). 22.5s → 6.9s (~3.2× over CPU); 11/11 cos≥0.999.

- **Implicit GEMM im2col-free conv** (`oxibonsai-image`, `src/vae/vae_conv_implicit.rs`): routed in `encode_conv2d_f32` for k≥3; eliminates im2col allocation for convolutions; VAE 9.1s → 6.9s.

- **Native VAE safetensors loader** (`oxibonsai-image`, `src/vae/safetensors.rs`): reads FLUX.2 `.safetensors` directly — bf16→f32 lossless decode (bit-shift, zero rounding error), conv weight transpose [O,I,kH,kW]→[O,kH,kW,I], `to_out.0` ModuleList un-nesting. Eliminates the Python `.npy` export step entirely. Parity gate: `vae_safetensors_parity` (bit-identity + determinism).

- **Qwen3-4B Text Encoder (4-bit)** (`oxibonsai-image`): `open_mlx_4bit` loads native 2.1 GB MLX 4-bit safetensors directly (was 15 GB f32 `.npy`). Dequantizes `mlx-packed-affine` 4-bit weights to f32 on demand. Real Bonsai-Image footprint ≈3.5 GB. Parity gate: `te_parity` cos≥0.999999 vs MLX oracle.

- **MLX-exact Threefry-2×32 RNG port** (`oxibonsai-image`, `src/sample/mlx_rng.rs`): byte-exact reproduction of MLX C++ (5 rounds, exact rotation constants and per-round key-inject, `primitives.cpp` fill layout). `--seed 42` byte-matches official mflux reference.

- **Flow-match Euler scheduler** (`oxibonsai-image`): dynamic μ-shift (seq_len-dependent, exponential time-shift type), native init noise, `img_ids`, `txt_ids`, sigmas/timesteps generation.

- **PNG output** (`oxibonsai-image`): oxiarc-deflate Pure-Rust DEFLATE (COOLJAPAN ecosystem; no flate2/zstd/zip). Parity-validated vs reference PNG at 512×512.

- **`oxibonsai image` CLI subcommand**: `oxibonsai image --prompt … --seed N --out x.png` standalone text-to-image command. Env vars: `OXI_DIT_GGUF`, `OXI_VAE_WEIGHTS` (dir OR `.safetensors`), `OXI_TE_4BIT`, `OXI_TE_TOKENIZER_DIR`.

- **CUDA imagen acceleration** (`oxibonsai-kernels`, `oxibonsai-image`): CUDA native-GPU backend for imagen on Linux/Windows (authored as a blind mirror of the Metal path; parity-first plain-FP32, cap-of-8-safe, additive — Metal byte-unchanged). Steps=4 benchmark on A4000-class GPU: ~101s → ~31.7s (3.2×); cos=1.0 throughout (FP32, no tensor cores). Three acceleration tiers:
  - **`gemm_tq2` kernel retile** (32×32/2×2 → 128×128/8×8 transposed-shared): ~6× kernel speedup.
  - **Flash-attention warp-cooperative** (lane-split head_dim + `__shfl_xor_sync`): ~6.3× kernel speedup.
  - **Stage-0 context_embedder GPU port** (`encode_gemm_f32`): 2.838s → 0.048s = 59× on that operation; DiT sample 6.17s → 3.32s; steps=4 45.4s → 31.7s.

- **Engine pool + CPU↔Metal parity guard** (`oxibonsai-runtime`, `src/engine_pool.rs`): `EnginePool` wrapping `Vec<InferenceEngine>` with `Arc<Semaphore>` permits. CPU tier = N replicas sharing one `Arc<[f32]>` token-embedding table (eliminates N duplicate ~1.16 GB allocations); GPU tier = clamped to 1 (Metal process-global singleton). `build_pool_from_gguf` API. `InferenceEngine` validates byte-identical greedy output across backends at startup. Default CPU pool size: `min(4, num_cpus)`.

- **`.env` / dotenvy auto-load**: all CLI subcommands now auto-load a `.env` file from the current directory or any parent directory (via `dotenvy`). Precedence: CLI flag > shell env > `.env` > built-in default. `.env.example` updated with all imagen-specific keys.

- **`docs/IMAGEN.md`** (new): end-to-end asset acquisition (DiT download → GGUF convert; TE 4-bit direct-load; VAE safetensors direct-load; tokenizer), environment setup, full CLI flag reference, build-per-GPU instructions, performance table (CUDA/Metal), parity validation harnesses, provenance declaration.

- **`docs/CLI.md`** (new): exhaustive flag and environment variable reference for all `oxibonsai` and `oxibonsai-serve` subcommands.

- **Stable+Nightly Rust build compatibility** (`oxibonsai-kernels`): previously required nightly Rust for `#[feature(stdarch_aarch64_prefetch)]`. Now builds cleanly on stable (1.86+) and nightly via:
  - `build.rs` nightly-detect: sets `cfg(nightly_aarch64_prefetch)` when nightly toolchain is detected.
  - `aarch64_prefetch!` macro: no-op on stable Rust, active prefetch on nightly (zero new warnings either way).

### Changed

- **Metal GPU default-on for imagen** (`oxibonsai-image`): DiT flash-attention and VAE GPU kernels are now default-on (opt-out via env vars); full pipeline ~52–62s on Apple Silicon M3 (steps=4, 512×512, vs ~10–15 min CPU-only). DiT sampler: 64.7s → 34.2s = 1.89×; VAE decode: 22.5s → 6.9s = 3.2×.

- **README.md**: Pure-Rust world-first declaration ("first pure-Rust, C/C++/Fortran-free, zero-FFI implementation of the Bonsai-Image FLUX.2-Klein text-to-image pipeline, built entirely on the COOLJAPAN ecosystem"); Documentation section linking `docs/IMAGEN.md` and `docs/CLI.md`.

- **`oxiarc-deflate`** workspace dependency: 0.3.1 → 0.3.2 (DEFLATE bug fix, affects PNG output correctness).

- **`serde_json`** workspace dependency: 1.0.149 → 1.0.150.

- **`tower-http`** workspace dependency: 0.6.10 → 0.6.11.

### Fixed

- **`fix(model): read head_dim from GGUF metadata for Qwen3-4B`** (`oxibonsai-model`): Qwen3-4B sets `head_dim=128` explicitly (`hidden_size / num_attention_heads = 2560 / 32 = 80 ≠ 128`). Deriving head_dim arithmetically caused `LinearTernary` shape mismatch (`expected [51200], got [81920]`) when loading Ternary-Bonsai-4B.gguf. Now reads `qwen3.attention.key_length` from GGUF metadata; falls back to arithmetic only when the key is absent.

- **`fix(completions): temperature-discard bug`** (`oxibonsai-runtime`, `src/completions.rs`): temperature was being silently discarded from `SamplingParams` on the streaming completions path. Now threaded correctly.

- **`fix(scripts): hf download arg order`** (`scripts/`): `hf` CLI argument order changed in newer releases; download scripts updated to match new `hf` CLI interface.

---

## [0.1.4] - 2026-05-16 - Phase 33

### Changed
- **block/types.rs split (policy compliance milestone)** (`oxibonsai-model`): split `crates/oxibonsai-model/src/block/types.rs` (1853 lines, 147-line margin) into a 10-file directory module `block/types/{mod, upload, layer_stats, scratch, block_def, forward_stats, helpers, forward_sw, forward, accessors}.rs`, max sub-file 424 lines. The single big `impl<'a> TransformerBlock<'a>` (107 methods) split across 5 sub-files: `block_def.rs` carries `pub struct TransformerBlock<'a>` + `new()`; `forward.rs` / `forward_stats.rs` / `forward_sw.rs` carry the three forward methods (390 / 264 / 306 lines respectively); `accessors.rs` carries all ~85 quant-family accessor methods (Q1, Ternary, FP8 E4M3/E5M2, Q4_0, Q8_0, Q2K/Q3K/Q4K/Q5K/Q6K/Q8K) grouped by family with comment dividers; `helpers.rs` carries `layer_idx` and the cfg-gated GPU/CUDA full-layer dispatch helpers. Visibility widened: 19 `TransformerBlock` fields + 16 `ScratchBuffers` fields from private to `pub(super)` so sibling sub-modules can destructure them; `try_full_layer_gpu` / `try_full_layer_cuda` from `fn` to `pub(super) fn`. All `block/mod.rs:23 pub use types::*;` public surface preserved verbatim — `LayerStats`, `TransformerBlock`, and all `pub fn` methods still accessible at `crate::block::*`. **Milestone**: after Phase 33, every Rust source file in the workspace has at least 227 lines of margin to the 2000-line per-file policy ceiling — the codebase has substantial headroom for future feature additions before any file reaches the limit again. New top-ranked Rust file is `cuda_k_quant_prefill_kernels.rs` at 1773 (kernel-source-string container, NVRTC C++ strings, not host Rust code). 4625 workspace tests pass; clippy clean.

---

## [0.1.4] - Phase 32

### Changed
- **forward_cuda.rs split (policy compliance)** (`oxibonsai-model`): split `crates/oxibonsai-model/src/model/types/forward_cuda.rs` (1801 lines, within 199 of the 2000-line per-file policy) into a directory module `forward_cuda/{mod.rs, byte_helpers.rs, q1.rs, ternary.rs, q_std.rs, k_quant.rs}`, mirroring the Phase 29/30 pattern. Max sub-file 580 lines (`q1.rs` carrying Q1 builders + 4 top-level dispatch entry points: `try_cuda_full_forward_inner`, `try_cuda_full_forward_with_lm_head`, `try_cuda_prefill_with_lm_head`, `try_cuda_prefill_verify`). The single big `impl<'a> BonsaiModel<'a>` block split across 5 sub-files (Rust allows multiple inherent impl blocks for the same type). Private inherent methods widened to `pub(super)` (helpers called from sibling sub-modules: `get_or_build_cuda_qkv_cache`, `build_cuda_layer_params`, `build_cuda_*_qkv_concats`, `build_cuda_*_layer_params`, intermediate dispatch helpers) or `pub(in super::super)` (entry points dispatched from `types/mod.rs`). Inner-attribute `#![cfg(all(feature = "native-cuda", any(target_os = "linux", target_os = "windows")))]` on `forward_cuda/mod.rs` eliminates 19 repeated per-method cfg attributes from the original. After Phase 32: every CUDA-related file that was at policy edge (cuda_prefill.rs, cuda_k_quant_prefill.rs, forward_cuda.rs) is now split. New top-ranked Rust source file is `block/types.rs` at 1853 (147-line margin). No behavioural changes; clippy clean; 4625 workspace tests pass.

---

## [0.1.4] - Phase 31

### Added
- **Cap-of-8 batch-kernel regression audit** (`oxibonsai-kernels`): new `tests/capof8_audit.rs` integration test enumerates every batch-GEMM kernel by name and asserts both that the kernel entry-point appears in its source string AND that the source contains the `col_base += 8` (CUDA) / `col_base += 8u` (MSL) outer-loop pattern. The pattern is mandatory for kernels that take a runtime `batch_size` argument — without it, any batch beyond 8 columns silently truncates (the `kernel_pattern_capof8` audit pattern documented from Phase 13.x prefill regressions). The test covers **36 CUDA batch kernels** (`cuda_prefill_kernels`: 6 Q1+TQ2 variants; `cuda_k_quant_prefill_kernels`: 18 across Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K; `cuda_fp8_prefill_kernels`: 6 FP8 E4M3/E5M2 variants; `cuda_q_std_prefill_kernels`: 6 Q4_0/Q8_0 variants) and **10 Metal MSL batch kernels** (`kernel_sources::prefill`: 4 Q1+TQ2; `kernel_sources::fp8_prefill`: 6 FP8 variants). Per-token GEMV kernels (`gemv_*_pf` and the standard decode-path GEMVs) are deliberately omitted — they take `batch_size = 1` by construction. Two host-only sanity tests verify the assertion helper itself rejects sources missing either the cap-of-8 pattern or the kernel name. Tests are feature-gated to match the kernel-source files (CUDA tests run on `--features native-cuda` on Linux/Windows; Metal tests run on `--features metal` on macOS). 4625 workspace tests passing (4619 + 4 CUDA audit + 2 host sanity).

---

## [0.1.4] - Phase 30

### Changed
- **Metal + runtime module refactor (policy compliance)** (`oxibonsai-kernels`, `oxibonsai-runtime`): Two more files within 60 lines of the 2000-line per-file policy were split into directory modules using the Phase 29 pattern. `metal_graph.rs` (1948 lines) → `metal_graph/{mod.rs, error.rs (67), reformat.rs (83), pipelines.rs (358), buffers.rs (152), graph.rs (626), tests.rs (670)}` — all public surface (`MetalGraph`, `MetalGraphError`, `MetalWeightHandle`) re-exported from `mod.rs`; internal helpers `alloc_buf`, `div_ceil`, `set_scalar`, `upload_f32`, `download_f32` re-exported `pub(crate)` for sibling Metal modules (`metal_dispatch`, `metal_fp8_kernels`, `metal_fp8_prefill`, `metal_full_layer/*`, `metal_prefill/*`); `MetalPipelines` kept `pub(crate)` (accessed via `self.pipelines.<kernel>` from `metal_dispatch`). `constrained_decoding.rs` (1966 lines) → `constrained_decoding/{mod.rs, error_trait.rs (106), regex.rs (724), json.rs (597), sampler.rs (217), allow_list.rs (123), sequence.rs (93), length.rs (128)}` — all 11 public types (`ConstraintError`, `TokenConstraint`, `NoConstraint`, `RegexConstraint`, `JsonConstraint`, `JsonParseState`, `ConstrainedSampler`, `ConstrainedSamplerBuilder`, `AllowListConstraint`, `SequenceConstraint`, `LengthConstraint`) re-exported from `mod.rs`; regex NFA internals (`NfaState`, `RegexNfa`, `Fragment`) tightened to `pub(super)` (needed only by nested `regex::tests`); all 33 inline tests preserved across the sub-modules. Combined with Phase 29: every file that was within 60 lines of the policy ceiling is now split; the new top-ranked Rust source file is 1853 lines (147 lines of margin). No behavioural changes; clippy clean; 4619 tests pass.

---

## [0.1.4] - Phase 29

### Changed
- **CUDA prefill module refactor (policy compliance)** (`oxibonsai-kernels`): Both `cuda_prefill.rs` (1989 lines, within 11 of the 2000-line per-file policy) and `cuda_k_quant_prefill.rs` (1941 lines) were split into directory modules so they no longer block additions on the per-file budget. The split mirrors the established `cuda_full_layer/` template. The new layout for **`cuda_prefill/`** is `mod.rs` + `state.rs` (types, singleton, init, buffer/KV/logits acquisition) + `launchers.rs` (7 launchers: Q1 GEMM + Q1 GEMM+residual + Q1 fused gate+up+SwiGLU + batched RMSNorm + 3 TQ2 variants) + `encode_q1.rs` (`encode_prefill_ffn_phase`, `encode_prefill_layer`) + `encode_ternary.rs` (`encode_prefill_ffn_phase_ternary`, `encode_prefill_layer_ternary`) + `try_apis.rs` (`try_cuda_prefill`, `try_cuda_prefill_ternary`) + `tests.rs`. The new layout for **`cuda_k_quant_prefill/`** is `mod.rs` + `state.rs` (`KQuantFormat`, `CudaKQuantPrefillModules`, `CudaKQuantPrefillLayerParams`, singleton, init, buffer/KV/logits acquisition) + `launchers.rs` (18 launchers: 3 per format × 6 K-quant formats) + `encode.rs` (`encode_k_quant_ffn_phase`, `encode_k_quant_prefill_layer` with format dispatch) + `try_api.rs` (`try_cuda_prefill_k_quant`) + `tests.rs`. All public surface preserved verbatim (`super::cuda_prefill::*` and `super::cuda_k_quant_prefill::*` access paths continue to work via top-level `pub use` re-exports). Sibling modules (`cuda_fp8_prefill`, `cuda_q_std_prefill`) that import `init_prefill_modules`, `CudaPrefillBuffers`, `CudaPrefillModules` from `super::cuda_prefill` are unaffected. Internal helpers (launchers, buffer acquisition) demoted from `unsafe fn` / `fn` to `pub(super) unsafe fn` / `pub(super) fn` for tighter scoping. No behavioural changes; 508 kernel-crate tests + 4619 workspace tests still pass; clippy clean.

---

## [0.1.4] - Phase 28

### Added
- **Metal FP8 batch GEMM primitives** (`oxibonsai-kernels`): 8 new MSL kernels mirror the Phase 26 CUDA FP8 batch prefill design on Apple Silicon Metal. `kernel_sources/fp8_prefill.rs` adds `MSL_GEMM_FP8_E4M3_V1`, `MSL_GEMM_FP8_E4M3_RESIDUAL_V1`, `MSL_FUSED_GATE_UP_SWIGLU_GEMM_FP8_E4M3_V1`, `MSL_GEMV_FP8_E4M3_PF_V1`, and their E5M2 counterparts. All batch kernels use column-major I/O (`buf[col * dim + element]`), AoS 34-byte FP8 blocks (`qs[32] + d:f16`), one simdgroup per output row, and the cap-of-8 outer-loop pattern (`for col_base in 0..batch_size step 8u`) so arbitrary batch sizes are processed correctly. Fused gate+up kernel reads gate rows `[0..n_ffn_rows)` and up rows `[n_ffn_rows..2*n_ffn_rows)` from a single concatenated FP8 weight buffer and emits `SiLU(gate_dot) * up_dot` per (row, col). New `gpu_backend/metal_fp8_prefill.rs` module compiles all 6 batch pipelines lazily into its own singleton (Device + CommandQueue, `OnceLock`-guarded, kept separate from Phase 27's GEMV singleton so processes that never touch prefill pay no init cost) and exposes 6 public host fns: `metal_gemm_fp8_e4m3` / `metal_gemm_fp8_e5m2`, `metal_gemm_fp8_e4m3_residual` / `metal_gemm_fp8_e5m2_residual`, `metal_fused_gate_up_swiglu_fp8_e4m3` / `metal_fused_gate_up_swiglu_fp8_e5m2`. 7 CI-GPU-gated CPU↔Metal parity tests (including a `batch_size = 12` cap-of-8 discriminator and a non-multiple-of-8 row-count boundary), plus host-only kernel-source-string assertions and shape-rejection tests. Forward-path integration (5-stage encoder, LM-head dispatch) is deferred to Phase 28.B.

---

## [0.1.4] - Phase 27

### Added
- **Metal FP8 GEMV** (`oxibonsai-kernels`, `oxibonsai-model`): first-class FP8 E4M3FN and E5M2 GEMV on Apple Silicon Metal GPUs. Two new MSL kernels (`MSL_GEMV_FP8_E4M3_V1`, `MSL_GEMV_FP8_E5M2_V1`) with AoS 34-byte blocks (`qs[32] + d:f16`), one simdgroup per output row, fully unrolled 32-weight inner loop, and `simd_sum` warp reduction. New `gpu_backend/metal_fp8_kernels.rs` module holds its own singleton (Device + CommandQueue + 2 pipelines) so `metal_graph.rs` stays under the 2000-line policy. Public host functions `metal_gemv_fp8_e4m3` / `metal_gemv_fp8_e5m2`. `impl Fp8Kernel for KernelDispatcher` now dispatches Metal → CUDA → CPU SIMD on the `Gpu` tier. `BonsaiModel::forward_prefill` and `forward_prefill_verify` skip the fused Metal prefill for FP8 models (Metal FP8 falls through to per-token sequential, which now uses Metal GEMV via the dispatcher). Includes 2 CI-GPU-gated CPU↔Metal parity tests + host-only MSL source-string assertions.

---

## [0.1.4] - Phase 26

### Added
- **CUDA FP8 E4M3/E5M2 batch prefill** (`oxibonsai-kernels`, `oxibonsai-model`): `try_cuda_prefill_fp8` replaces sequential single-token CUDA GEMV loop for FP8E4M3 and FP8E5M2 models. 8 new NVRTC kernels (gemm, gemm_residual, fused_gate_up_swiglu, gemv_pf × E4M3/E5M2) with cap-of-8 outer loop and correct FP8 block layout (weights at bytes 0-31, FP16 scale at bytes 32-33). New `forward_cuda_fp8.rs` module with `try_cuda_prefill_with_lm_head_fp8` / `try_cuda_prefill_verify_fp8`; handle namespaces E4M3 26M-31M, E5M2 28M-33M. `BonsaiModel::forward_prefill` and `forward_prefill_verify` now route FP8 models through batch GEMM (with sequential CUDA GEMV fallback on error). 4 host-only kernel-source-string tests added.

---

## [0.1.4] - Phase 25

### Added
- **K-quant CUDA batch prefill** (`oxibonsai-kernels`, `oxibonsai-model`): `try_cuda_prefill_k_quant` replaces sequential single-token CUDA GEMV loop for Q2K, Q3K, Q4K, Q5K, Q6K, and Q8K models. 18 new NVRTC kernels (3 per format: `gemm_q{fmt}`, `gemm_q{fmt}_residual`, `fused_gate_up_swiglu_gemm_q{fmt}`) with cap-of-8 outer loop preventing silent batch-size truncation. `KQuantFormat` enum for dispatch. `BonsaiModel::forward_prefill` and `forward_prefill_verify` now route K-quant models through `try_cuda_prefill_with_lm_head_k_quant` / `try_cuda_prefill_verify_k_quant` on CUDA hosts (with sequential fallback on error). Handle namespaces: Q2K norms `12M`, weights `13M`; Q3K `14M/15M`; Q4K `16M/17M`; Q5K `18M/19M`; Q6K `20M/21M`; Q8K `22M/23M`; final-norm `24M+fmt_offset`; LM-head `25M+fmt_offset`. 42 K-quant block accessor methods added to `TransformerBlock`. Runtime validation: `hidden_size % 256 == 0` required for all K-quant batch GEMM.

---

## [0.1.4] - Phase 24

### Added
- **CUDA Q4_0/Q8_0 batch prefill** (`oxibonsai-kernels`, `oxibonsai-model`): `try_cuda_prefill_q_std` replaces sequential single-token prefill with fused batch GEMM for Q4_0 and Q8_0 models. `BonsaiModel::forward_prefill` now routes Q4_0/Q8_0 through `try_cuda_prefill_with_lm_head_q_std` (with sequential fallback on error). `try_cuda_prefill_verify` routes Q4_0/Q8_0 through `try_cuda_prefill_verify_q_std`. Handle namespaces: norm `8M/10M`, weight `9M/11M` (Q4_0/Q8_0 respectively), final-norm `8.9M/10.9M`, LM-head `9.9M/11.9M`.

---

## [0.1.4] - Phase 23

### Added
- **K-quant CUDA GEMV kernels** (`oxibonsai-kernels`, `oxibonsai-model`): Six new NVRTC GEMV kernels for Q2K, Q3K, Q4K, Q5K, Q6K, and Q8K K-quant formats. `LinearQ2K/Q3K/Q4K/Q5K/Q6K/Q8K::forward` now dispatch to `cuda_gemv_q*k` NVRTC kernels when a CUDA device is available (Linux/Windows, `native-cuda` feature), falling back to CPU scalar on error. `BonsaiModel::forward` and `forward_prefill` detect `OutputWeight::Q2K/Q3K/Q4K/Q5K/Q6K/Q8K` + CUDA availability and route through the block dispatch loop, bypassing the Q1-only CUDA graph attempt. Compile-time `Block*` layout assertions added.

---

## [0.1.4] - Phase 22

### Added
- **Q4_0/Q8_0 CUDA full-forward dispatch** (`oxibonsai-model`, `oxibonsai-kernels`): `LinearQ4_0::forward` and `LinearQ8_0::forward` now dispatch to `cuda_gemv_q4_0`/`cuda_gemv_q8_0` NVRTC kernels when a CUDA device is available (Linux/Windows, `native-cuda` feature), falling back silently to CPU scalar on error. `BonsaiModel::forward` and `forward_prefill` detect `OutputWeight::Q4_0/Q8_0` + CUDA availability and route directly through the block dispatch loop, bypassing the Q1-only CUDA graph attempt. Includes `tracing::debug!` profile logging and compile-time `BlockQ4_0`/`BlockQ8_0` layout assertions.

---

## [0.1.4] - Phase 21

### Added
- **FP8 CUDA early-dispatch** (`oxibonsai-model`): `BonsaiModel::forward` and `forward_prefill` now detect `OutputWeight::FP8E4M3/E5M2` + CUDA availability and route directly through the block dispatch loop (which calls `cuda_gemv_fp8_e4m3/e5m2` via `KernelTier::Gpu`), bypassing the Q1-only CUDA graph attempt entirely. Includes `tracing::debug!` profile logging.
- **Block accessor methods** (`oxibonsai-model`): 14 FP8 E4M3/E5M2 block accessors (`attn_q/k/v/output_blocks_fp8e4m3`, `ffn_gate/up/down_blocks_fp8e4m3`, same for E5M2) and 14 Q4_0/Q8_0 block accessors added to `TransformerBlock`, delegating to the existing `LinearLayer::blocks_*` methods.
- **CUDA Q4_0 / Q8_0 GEMV kernels** (`oxibonsai-kernels`): `cuda_gemv_q4_0` and `cuda_gemv_q8_0` NVRTC kernels with warp-per-row reduction. Q4_0: 18-byte AoS blocks (FP16 scale + nibble-packed int4); Q8_0: 34-byte AoS blocks (FP16 scale + int8). Public functions callable from the forward dispatch in a future phase.

---

## [0.1.4] - Phase 20

### Added
- **CUDA ternary batch prefill** (`oxibonsai-kernels`): three new NVRTC kernels (`gemm_tq2_g128_v7`, `gemm_tq2_g128_v7_residual`, `fused_gate_up_swiglu_gemm_tq2`) enabling fused TQ2_0_g128 batch GEMM on CUDA. Replaces the sequential single-token fallback for Ternary models on CUDA hosts. Cap-of-8 outer loop prevents silent batch-size truncation (kernel_pattern_capof8). New `try_cuda_prefill_ternary` entry point; per-token `encode_attn_phase_tq2` for sequential attention within the batched prefill.
- **CUDA FP8 GEMV** (`oxibonsai-kernels`): `cuda_gemv_fp8_e4m3` and `cuda_gemv_fp8_e5m2` NVRTC kernels with warp-per-row warp-shuffle reduction and AoS block decode (32-weight blocks, FP16 scale at byte offset 32). `KernelDispatcher` with `KernelTier::Gpu` now routes `Fp8Kernel::gemv_fp8_e4m3/e5m2` calls to the GPU path on Linux/Windows.

---

## [0.1.4] - Phase 15 (0.1.5 preview)

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

[0.2.3]: https://github.com/cool-japan/oxibonsai/releases/tag/v0.2.3
[0.2.2]: https://github.com/cool-japan/oxibonsai/releases/tag/v0.2.2
[0.2.1]: https://github.com/cool-japan/oxibonsai/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/cool-japan/oxibonsai/compare/v0.1.5...v0.2.0
[0.1.2]: https://github.com/cool-japan/oxibonsai/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/cool-japan/oxibonsai/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/cool-japan/oxibonsai/releases/tag/v0.1.0
