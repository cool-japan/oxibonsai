# OxiBonsai
# (オキシ盆栽)

**Pure Rust Sub-2-Bit LLM Inference Engine for PrismML Bonsai Models**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.89%2B-orange.svg)](https://www.rust-lang.org)

OxiBonsai is a zero-FFI, zero-C/C++ inference engine for PrismML's sub-2-bit Bonsai family: the **1-bit** line (Q1\_0\_g128), the **ternary** line (TQ2\_0\_g128) and the **Bonsai 2 27B** hybrid model (PTQ1\_0 / PQ2\_0 / group-64 Q2\_0), plus the Bonsai-Image text-to-image model. It runs on CPU (SIMD) and Apple Silicon (Metal) without depending on llama.cpp, BLAS, or any C/Fortran runtime; an NVIDIA (CUDA, `native-cuda`) backend has been validated on one GPU, an RTX A4000, for the 1-bit and ternary models (scope and gaps in [Known Limitations](#known-limitations)). Built on the COOLJAPAN ecosystem, it delivers sovereign AI inference in Pure Rust.

To our knowledge, OxiBonsai is the first pure-Rust — C/C++/Fortran-free, zero-FFI — inference engine for the Bonsai 1-bit/ternary model family, and the first to bring its FLUX.2-Klein text-to-image (Bonsai-Image) to pure Rust, built entirely on the COOLJAPAN ecosystem.

## Documentation

- [Model support matrix](docs/models.md) — every supported model and file, with exact ggml type ids, block layouts, sizes, and how the three layouts that share ggml type id 42 are told apart.
- [CLI reference](docs/CLI.md) — every `oxibonsai` and `oxibonsai-serve` subcommand, flag, and environment variable.
- [Deployment guide](docs/DEPLOYMENT.md) — which binary to run in production, the middleware order, reverse-proxy/TLS setup, bearer auth, admission limits, Prometheus scraping, checksum verification and the release checklist.
- [Image-generation guide](docs/IMAGEN.md) — end-to-end Bonsai-Image (FLUX.2-Klein) text-to-image walkthrough.
- [Changelog](CHANGELOG.md) — what changed in each release, including the behaviour changes to read before upgrading.
- [Known Limitations](#known-limitations) — honest current-reality notes on what is wired, what is experimental and what is deferred.

## Status

**Version 0.2.4** (unreleased) · ~401k lines of Rust code (~439k lines of code across all languages, per `tokei`) · Pure Rust

| Crate | Role |
|-------|------|
| oxibonsai-core      | GGUF reader/writer, quantization block types, model configuration |
| oxibonsai-kernels   | CPU SIMD tiers (scalar / AVX2 / AVX-512 / NEON, opt-in INT8), Metal and CUDA backends |
| oxibonsai-model     | Dense Qwen3 and hybrid Qwen3.5 forward passes, weight loaders, converters, vision tower |
| oxibonsai-runtime   | Inference engine, sampling, OpenAI-compatible server, chat template, reasoning and tool calls |
| oxibonsai-tokenizer | Pure-Rust BPE tokenizer, GGUF vocabulary reader, Jinja chat-template engine |
| oxibonsai-rag       | RAG pipeline (chunking, embedders, vector store) |
| oxibonsai-eval      | Evaluation harness (ROUGE, BLEU, chrF, perplexity, MMLU and other logit tasks) |
| oxibonsai-serve     | Standalone server binary |
| oxibonsai-image     | Bonsai-Image (FLUX.2 Klein) text-to-image: text encoder, DiT, VAE, PNG |
| oxibonsai (facade)  | One dependency re-exporting the crates above behind feature flags |
| oxibonsai-cli       | The `oxibonsai` binary (root package) |

`scripts/ci.sh` is the quality gate (fmt, builds, clippy `-D warnings`, nextest, doctests, rustdoc `-D warnings`, cargo-deny, the Pure-Rust policy check, wasm32, coverage baseline); `scripts/release-gate.sh` adds the real-model legs. Its CUDA kernel-syntax stage needs `nvcc`: on a release host without the CUDA toolkit the owner must pass `--accept-approximate-cuda-syntax`, an explicit waiver that is named in the run's verdict (see [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)). The measured test totals of the last full run are recorded in the release checklist, not hard-coded here.

## Features

### Sub-2-Bit Native Inference

Native quantization families, each with dedicated dequant / GEMV / full-forward kernels:

| Family | Encoding | Stored bits/weight | Block size | Example models |
|--------|----------|--------------------|------------|----------------|
| **1-bit**  | Q1\_0\_g128  | 1.125 (1 bit + scale) | 128 weights / 18 B, FP16 group scale first | Bonsai-8B, Bonsai-27B |
| **Ternary** | TQ2\_0\_g128 | 2.125 (≈1.585 bits of information) | 128 weights / 34 B, FP16 scale last | Ternary-Bonsai-8B / 4B / 1.7B |
| **Bonsai 2 ternary** | PQ2\_0 | 2.125 | 128 weights / 34 B, FP16 scale first | Ternary-Bonsai-2-27B |
| **Bonsai 2 1.75-bit** | PTQ1\_0 | 1.75 | 128 weights / 28 B, base-3 trits, FP16 scale last | Ternary-Bonsai-2-27B |
| **Bonsai 2 g64** | Q2\_0 (ggml id 42) | 2.25 | 64 weights / 18 B | Ternary-Bonsai-2-27B (fork-required file) |

- Dense Qwen3 (GQA, SwiGLU, RoPE incl. YaRN, RMSNorm, optional sliding window) and the **Qwen3.5 hybrid** of Bonsai 2 (Gated DeltaNet linear attention + gated full attention, Hadamard-rotated weights, M-RoPE)
- TQ2\_0\_g128 code map: `0b00→−1, 0b01→0, 0b10→+1, 0b11→0`. PQ2\_0 differs at `0b11` (`+2`; ternary data never emits it) and stores its scale first — see [`docs/models.md`](docs/models.md)
- Correctness gates: at `--temperature 0 --seed 42`, CPU and Metal produce byte-identical output on the real Ternary-Bonsai-1.7B; on the two Bonsai 2 27B files present on the reference host, greedy token ids equal the PrismML llama.cpp fork on CPU and Metal

### Acceleration Tiers

| Tier | Target | Width / Device | Feature Flag |
|------|--------|----------------|--------------|
| Reference | All platforms | Scalar | (default) |
| AVX2 + FMA | x86-64 | 256-bit | `simd-avx2` |
| AVX-512 | x86-64 | 512-bit | `simd-avx512` |
| NEON | AArch64 | 128-bit | `simd-neon` |
| **Metal** | Apple Silicon | GPU, fused full-forward (dense and hybrid runner) | `metal` |
| **CUDA (native)** | NVIDIA GPU | GPU, NVRTC kernels — validated on one RTX A4000 (CUDA 12.0); no Bonsai 2 27B forward | `native-cuda` |
| **CUDA (scirs2)** | NVIDIA GPU | CPU SIMD fallback † | `cuda` |

> † **The `cuda` (scirs2-core) tier runs on CPU, not GPU.** scirs2-core retired its cudarc-based CUDA backend in 0.6.x, so `is_accelerated()` always returns `false` for this tier and the dispatcher transparently falls back to CPU SIMD — output is correct, just not GPU-accelerated. Use `native-cuda` for NVIDIA GPU acceleration. It has been run on one NVIDIA RTX A4000 (Ampere, compute capability 8.6), CUDA 12.0, x86_64 Linux, on 2026-10-07: CPU↔CUDA parity for Bonsai-8B, Ternary-Bonsai-1.7B/8B and Q4_0 / Q8_0 / K-quant / FP8 test fixtures. Other GPU generations, aarch64 Linux, Windows and multi-GPU hosts have not been run, and the Bonsai 2 hybrid kernels have no CUDA forward pass. A `CudaBackend` / `MetalBackend` stub no longer exists; the `cuda` feature is the scirs2 abstraction tier, not an NVIDIA build.

Auto-detection via `KernelDispatcher::auto_detect()` selects the best CPU tier at runtime. GPU backends are opt-in at build time (`--features metal` / `native-cuda`) and chosen per run with `--backend auto|cpu|metal` (`auto`, the default, picks the best tier for the build and host).

> **Note on CPU tiers:** The CPU tier is chosen *entirely at runtime* via `is_x86_feature_detected!` — the dispatcher picks AVX-512 only when AVX-512F+BW+VL are all present, otherwise AVX2+FMA, otherwise the scalar reference path. Each SIMD function carries a per-function `#[target_feature(...)]` attribute, so a single x86-64 binary is safe on every x86-64 CPU and automatically falls back (AVX-512 → AVX-2 → scalar) with no SIGILL. The `simd-avx2` / `simd-avx512` / `simd-neon` Feature Flags above are accepted for compatibility but do **not** gate tier selection — all tiers are always compiled in and chosen at runtime.
>
> AVX-512 has been absent from Intel *consumer* CPUs since Alder Lake (Raptor Lake, Meteor Lake, Arrow Lake and Lunar Lake have none); it mainly benefits Xeon / HEDT and AMD Zen 4+. On consumer hardware the AVX-2 tier is selected automatically.
>
> **Opt-in INT8 dot-product tier.** Setting `OXIBONSAI_KERNEL_TIER` to `int8-scalar`, `neon-int8`, `neon-dot`, `neon-i8mm` or `avx512-vnni` runs the CPU GEMV/GEMM of the native `TQ2_0_g128` / `Q1_0_g128` formats and the PrismML `PQ2_0` / group-64 `Q2_0` formats on INT8 kernels: the activation is quantized to INT8 once per call (one scale per weight block) and every block accumulates exactly in `i32` (`SDOT` on `neon-dot`; `SMMLA` for batched GEMM plus `SDOT` for GEMV on `neon-i8mm`; `VPDPBUSD` on `avx512-vnni`). It is **never selected by default** — unset, the f32 kernels run bit-identically — and it is lossy but bounded: the selector reaches the native TQ2\_0\_g128 / Q1\_0\_g128 CPU tiers at all seven kernel entry points (minimum cosine against f32 0.999991–0.999993 on real weights; `neon-i8mm` decodes the 1.7B on the CPU 8.76× faster), and the PrismML formats honour it on any backend for the CPU model. The GPU tier is never diverted — Metal output is byte-identical with the variable set, proven on the real 1.7B and Bonsai-8B — **while the GPU prefill actually runs**: when a GPU prefill is declined or fails, the CPU batched prefill (and the batched CPU embedding pass, `forward_hidden`, on any engine) runs on the selected INT8 tier, so an operator must not read "Metal output unchanged" as "embeddings unchanged". `PTQ1_0` has no INT8 kernel, and `avx512-vnni` compiles but has never run on VNNI hardware.

### Fused GPU Full-Forward Path

Both the 1-bit and ternary forward passes are encoded into a **single GPU command buffer** rather than one submission per GEMV. Per-layer dispatch sequence:

1. Pre-attn RMSNorm
2. Fused QKV GEMV (Q ‖ K ‖ V concatenated in weight SoA)
3. Fused QK-norm + RoPE
4. Fused KV-store
5. Batched attention: scores V2 → softmax → weighted-sum
6. Attn output GEMV + residual add
7. FFN RMSNorm
8. Gate + Up GEMV (gate ‖ up concatenated)
9. Batched SwiGLU
10. Down GEMV + residual add

= 14 dispatches/layer × N layers per command buffer. This is what unlocks the Metal throughput numbers below.

The **Bonsai 2 27B** runs on a separate **Metal hybrid runner** (Gated-DeltaNet recurrence, causal conv1d, blockwise Hadamard transform and PQ2\_0 / PTQ1\_0 GEMV kernels in one command buffer per token). With `--backend auto` on a Metal host the 27B runs there, with its weights mapped in place (no copy), and `--backend cpu` keeps the CPU tier; `--backend metal` refuses with a typed error when the build or host cannot serve the model. With a vision projector (`--mmproj`) the vision tower also runs on the GPU and the runner prefills the image rows.

Dense prefill on Metal was super-linear in the batch size before 0.2.4 (the row-wise 1-bit GEMM re-read its input once per weight row); a simdgroup-tiled GEMM replaced it from 8 rows, so fused prefill is 8× faster per token than sequential decode at 256 tokens on Bonsai-8B (numbers in [Measured Throughput](#measured-throughput)).

### Observability

- Structured logging via `tracing` with env-filter and JSON output; the selected kernel tier is logged once per process per (tier, reason)
- Inference metrics: tokens/sec, prefill/decode latency, request counts (`/metrics`)
- `/health` is an unauthenticated liveness probe; `/readyz` reports readiness: `200` while a model is loaded and the engine pool can serve, **including while every replica is busy and the server is at its concurrency limit** (a saturated server is ready-but-saturated; `engine_slot_available` in the body says whether a replica is idle right now), and `503 not_ready` only for a server that cannot serve at all. `/readyz` is not exempt from bearer auth (see [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md#health-readiness-and-metrics)). `/health`, `/readyz` and `/metrics` are answered outside the concurrency budget, so they keep answering while the server is at its limit; `GET /v1/models` and the API routes are inside it and answer `503` with `Retry-After` there
- **Per-request tracing IDs** via `RequestId` (RFC 4122 UUIDv4, no `uuid` crate dependency)
- **Per-request rate metrics** via `RequestRateTracker` — TBT p50/p95, EWMA tokens/sec, queue-wait — and a `RequestRateAggregator` that rolls snapshots into the `oxibonsai_request_tokens_per_second`, `oxibonsai_inter_token_latency_p50/p95_seconds` and `oxibonsai_queue_wait_seconds` gauges. These are **library APIs**: the shipped server binaries do not attach an aggregator, so those four gauges read 0 there unless an embedder attaches one with `InferenceEngine::set_rate_aggregator`.
- A `CircuitBreaker` and a `HealthReport` exist as library modules (`oxibonsai-runtime`); neither is wired into the shipped servers. Overload protection in the servers is the admission layer (`--max-concurrent-requests`, capped at 4 × the engine-pool size → 503 with `Retry-After`; `--request-timeout-ms` → a stage-naming 504 from the three generation routes — chat, extended chat and completions — and from `/rag/query` when `--rag` mounts it, which also cancels the generation, as does a client that disconnects; the admission layer's 408 remains a backstop for the other routes; both carry `error.code: request_timeout`) and optional per-IP rate limiting (`--rate-limit-rpm` → 429).

### Runtime Controllers (0.1.4)

Two adaptive controllers shipped in 0.1.4 let the runtime self-tune as the workload changes:

```rust
use oxibonsai_runtime::{KvCachePolicy, AdaptiveLookahead, AdaptiveLookaheadConfig};

// KV cache policy: computes a target FP16 / Q8 / Q4 level from EWMA pressure
// with hysteresis. NOTE: this is advisory/telemetry only today — `observe()`
// drives the `/admin/*` JSON endpoint and a Prometheus gauge, it does not
// itself change the model's actual KV-cache storage precision (see
// "Known Limitations").
let kv = KvCachePolicy::default();
let level = kv.observe(0.92);  // → target escalates to Q8 once smoothed pressure crosses 0.80

// Speculative-decoding draft length: continuously updated from acceptance EWMA.
let mut k = AdaptiveLookahead::new(AdaptiveLookaheadConfig::default());
k.observe_step(5, 4);  // proposed=5, accepted=4 → k drifts toward 5
```

A worked end-to-end example lives in `examples/runtime_controllers.rs`:

```bash
cargo run --example runtime_controllers
```

### OpenAI-Compatible API

- `/v1/chat/completions` (POST), including streaming SSE, tool calls (streamed too), `reasoning_content` for `<think>` blocks, `enable_thinking` / `reasoning_effort`, per-request `seed` / `min_p` / penalties / `logprobs`, and — with a vision projector — `image_url` content parts: base64 data URIs, `file://` inside `--media-path`, and `http(s)` URLs only with `--allow-image-url-fetch` (public addresses only, plus the hosts `--image-url-allow-host` names; one deadline per image, `--image-url-timeout-ms`)
- `/v1/chat/completions/extended`, `/v1/completions` (batched prompts, streaming, `logprobs`), `/v1/embeddings` (dense and hybrid models; or a TF-IDF backend), `/v1/models`
- The prompt is rendered through the **GGUF's own `tokenizer.chat_template`** (a real Jinja subset), with a ChatML/Qwen3 fallback for files that ship none — it is not a hardcoded ChatML string
- Bearer auth, admission control, per-IP rate limiting, CORS, body limit and `/admin/*` (token-gated) — see [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)

### Builder Pattern API

```rust
use oxibonsai_runtime::{EngineBuilder, SamplingPreset};

let engine = EngineBuilder::new()
    .model_path("models/Ternary-Bonsai-1.7B.gguf")
    .preset(SamplingPreset::Balanced)
    .max_seq_len(4096)
    .build()?;
```

### Sampling Presets

| Preset | Temperature | Top-K | Top-P | Use Case |
|--------|-------------|-------|-------|----------|
| Greedy | 0.0 | 1 | 1.0 | Deterministic |
| Balanced | 0.7 | 40 | 0.9 | General |
| Creative | 1.0 | 100 | 0.95 | Creative writing |
| Code | 0.2 | 10 | 0.8 | Code generation |

The CLI and server take defaults from the model file's own `general.sampling.*` keys when present (Bonsai 2: temperature 1.0, top-p 0.95, top-k 20), then fall back to 0.7 / 0.9 / 40. No repetition penalty is applied unless one is requested.

## Bonsai Model Family

OxiBonsai supports PrismML's Bonsai lineup across both quantization families and the Bonsai 2 hybrid. The full matrix — exact ggml type ids, byte layouts, sizes, repositories, memory plan — is in [`docs/models.md`](docs/models.md); the summary:

| Model | Arch | Format | Size | Context |
|-------|------|--------|------|---------|
| Bonsai-8B                | `qwen3` (36 layers) | Q1\_0\_g128  | 1.16 GB  | 65,536 (YaRN ×4 over 16,384) |
| Ternary-Bonsai-8B        | `qwen3` (36 layers) | TQ2\_0\_g128 | 2.18 GB  | 65,536 |
| Ternary-Bonsai-4B        | `qwen3` (24 layers) | TQ2\_0\_g128 | —        | 65,536 |
| Ternary-Bonsai-1.7B      | `qwen3` (28 layers) | TQ2\_0\_g128 | 540 MB   | 32,768 |
| **Ternary-Bonsai-2-27B** | `qwen35` hybrid (64 layers) | PQ2\_0 / PTQ1\_0 / Q2\_0 g64 | 7.21 / 5.95 / 7.63 GB | 262,144 |
| Bonsai-27B, Ternary-Bonsai-27B (previous generation) | `qwen35` hybrid | Q1\_0 / PQ2\_0 / Q2\_0 | 3.80 / 7.17 / 7.17 GB | 262,144 |

The dense models share the Qwen3 architecture, so the runtime, tokenizer, and server are identical across that family; the Bonsai 2 27B adds the hybrid stack described above. The built-in dense configurations equal the real file headers (they did not before 0.2.4: the old 1.7B and 8B numbers matched no shipped file).

> **Note:** PrismML publishes Ternary Bonsai 8B / 4B / 1.7B as unpacked safetensors. Use `scripts/download_ternary.sh` (or `oxibonsai convert --quant tq2_0_g128`) to fetch and repack as GGUF before loading; `oxibonsai pull` downloads the 27B files, the vision projector and Bonsai-8B directly. An `onnx-community` ONNX release (MatMulNBits bits=2) is also supported via `oxibonsai convert --onnx`.

**Model behaviour you may notice.** Models that declare `rope.scaling.type = yarn` (Bonsai-8B: factor 4, original context 16,384) now decode with YaRN applied at **all** positions (mscale 1.1386), matching llama.cpp, so greedy output differs from OxiBonsai ≤ 0.2.3 on such models; the new output is byte-identical to the PrismML llama.cpp fork. `--rope-scaling off` reproduces the old behaviour.

## Installation

### CLI (recommended for end users)

```bash
cargo install oxibonsai-cli                    # CPU build
cargo install oxibonsai-cli --features metal   # Apple Silicon GPU build
```

This installs the `oxibonsai` binary. Rust 1.89+ required.

### Library (for Rust projects)

```toml
[dependencies]
oxibonsai = "0.2.4"
```

### Build from source (for development)

```bash
git clone https://github.com/cool-japan/oxibonsai
cd oxibonsai
cargo build --release
# binary at: target/release/oxibonsai
```

## Configuration (`.env`)

The CLI auto-loads a `.env` file from the current directory (or any parent), so you can
omit the model/path flags. Precedence: `--flag` > `--config` file > shell env var > `.env` > built-in default (see [`docs/CLI.md`](docs/CLI.md#precedence)).

How the file is loaded (only the `oxibonsai` binary does this; `oxibonsai-serve` reads no `.env`): `dotenvy` walks up from the working directory and loads the **first** `.env` it finds, setting every `KEY=value` in it as a process environment variable **unless that variable is already set** — so an exported variable always wins over the file, and any variable listed under [Environment variables](docs/CLI.md#environment-variables) can live there (the `OXI_*` model and image-pipeline paths and toggles, the `OXIBONSAI_*` server and runtime settings). Because the lookup continues into parent directories, **a stale `.env` above a checkout is loaded by every run beneath it and silently changes what those runs do** (an old `OXI_DIT_GGUF` pointing `oxibonsai image` at a model you have since moved, say), and the binary does not print which file it loaded. If a run behaves unexpectedly, look for a `.env` in the working directory and each parent (`ls -a . ..`) and compare it with `env | grep '^OXI'`; the template [`.env.example`](.env.example) has every entry commented out for that reason.

```bash
# Fetch the template from GitHub …
curl -fsSL https://raw.githubusercontent.com/cool-japan/oxibonsai/master/.env.example -o .env
# … or, in a source checkout:  cp .env.example .env

# Edit .env to point at your model files
$EDITOR .env
```

Keys:

| Key | Used by | Purpose |
|-----|---------|---------|
| `OXI_MODEL` | `run` / `chat` / `serve` / `info` | GGUF model path (omit `--model`) |
| `OXI_TOKENIZER` | `run` / `chat` / `serve` | tokenizer.json/dir (optional) |
| `OXI_DIT_GGUF` | `image` | FLUX.2 Klein ternary DiT GGUF |
| `OXI_VAE_WEIGHTS` | `image` | VAE decoder weights dir |
| `OXI_TE_4BIT` | `image` | 2.1 GB 4-bit MLX text-encoder `model.safetensors` |
| `OXI_TE_TOKENIZER_DIR` | `image` | text-encoder tokenizer dir |
| `OXI_DIT_ATTN_GPU`     | `image` / `repl` | Enable Metal/CUDA DiT flash-attention (default: on for Metal) |
| `OXI_VAE_GPU`          | `image` / `repl` | Enable Metal/CUDA VAE decode (default: on for Metal) |
| `OXI_TE_GPU`           | `image` / `repl` | Enable GPU text-encoder (experimental; default off) |

With `.env` in place, the flags become optional:

```bash
oxibonsai run   --prompt "Explain ternary quantization in one sentence."
oxibonsai image --prompt "a tiny bonsai tree in a ceramic pot" --out bonsai.png
```

## Quick Start

> **If you installed via `cargo install oxibonsai-cli`**, start from Step 2.
> The `oxibonsai` binary is already on your PATH.

### Step 1 — (source builds only) Build

```bash
cargo build --release            # add --features metal on Apple Silicon
export PATH="$PWD/target/release:$PATH"
```

### Step 2 — Get a model

Pick **one** of the families (or grab several):

```bash
# ── Option A: 1-bit Bonsai-8B (1.16 GB pre-quantized GGUF) ───────────────
oxibonsai pull bonsai-8b                       # verified against a compiled-in SHA-256

# ── Option B: Bonsai 2 27B (hybrid; ~7.2 GB PQ2_0 band, ~5.9 GB with --band ptq1) ─
oxibonsai pull bonsai2-27b                     # add --vision for the mmproj projector

# ── Option C: Ternary Bonsai (download safetensors + convert to GGUF) ────
# Fetches unpacked safetensors from HF and runs `oxibonsai convert`
# to produce models/Ternary-Bonsai-<size>.gguf + models/tokenizer.json.
./scripts/download_ternary.sh 1.7b    # also: 4b | 8b
```

> **Ternary prerequisite:** `scripts/download_ternary.sh` uses the
> HuggingFace `hf` CLI — install with `pip install huggingface_hub`.

### Step 3 — Get the tokenizer

The Bonsai 2 GGUFs carry their own vocabulary and chat template, so no separate tokenizer
file is needed for them. For the other models:

```bash
oxibonsai tokenizer download          # saves to models/tokenizer.json
```

The tokenizer is pulled from `Qwen/Qwen3-8B` on HuggingFace (~2.7 MB).
Use `--output` to save elsewhere, `--repo` to use a different HF repo.
Option C above already downloads it automatically.

### Step 4 — Run inference

> **Tip:** set `OXI_MODEL` (and optionally `OXI_TOKENIZER`) in `.env`
> (see [Configuration](#configuration-env)) to omit `--model`.

```bash
# 1-bit Bonsai-8B
oxibonsai run --model models/Bonsai-8B.gguf \
  --prompt "Explain quantum computing in simple terms" \
  --max-tokens 512 --temperature 0.7 --top-p 0.9

# Bonsai 2 27B through its own chat template, with reasoning shown on stderr
oxibonsai run --model models/Ternary-Bonsai-2-27B-PQ2_0.gguf --chat \
  --reasoning-effort low --prompt "Why is the sky blue?"

# Bonsai 2 27B with an image (needs the mmproj projector)
oxibonsai run --model models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  --mmproj models/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf \
  --image photo.png --prompt "Describe this image." --max-tokens 128

# ... or an image by URL: remote images are fetched only when opted in, from
# public addresses only (see docs/CLI.md, "Remote images")
oxibonsai run --model models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  --mmproj models/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf --allow-image-url-fetch \
  --image https://example.com/photo.png --prompt "Describe this image."

# Ternary Bonsai (same CLI, different file)
oxibonsai run --model models/Ternary-Bonsai-1.7B.gguf \
  --prompt "Explain quantum computing in simple terms" \
  --max-tokens 512 --temperature 0.7 --top-p 0.9

# Interactive chat, model info, server — all model-agnostic:
oxibonsai chat   --model models/Bonsai-8B.gguf
oxibonsai info   --model models/Ternary-Bonsai-1.7B.gguf
oxibonsai serve  --model models/Ternary-Bonsai-1.7B.gguf \
                 --host 127.0.0.1 --port 8080
oxibonsai build-info                           # features, kernel tiers, git commit

# Interactive image REPL — loads DiT/VAE/TE once, renders many prompts
oxibonsai repl   --seed 42 --steps 4 --width 512 --height 512

# Convert safetensors → GGUF (HuggingFace unpacked safetensors dir)
oxibonsai convert \
  --from <unpacked-safetensors-dir> \
  --to models/my-model.gguf \
  --quant tq2_0_g128        # default: native TQ2_0_g128 (id 42); also q1_0_g128, ptq1_0, q2_0_g64
# --quant pq2_0 writes PrismML PQ2_0 (id 142) for the PrismML llama.cpp fork;
# a qwen3 model converted that way does not load in 0.2.4 (no PQ2_0 LM-head path)

# Convert ONNX → GGUF (MatMulNBits bits=2, e.g. onnx-community/Ternary-Bonsai-1.7B-ONNX)
oxibonsai convert --onnx \
  --from path/to/model.onnx \
  --to models/my-model.gguf
```

## CLI Smoke & Benchmark Scripts

Two parallel smoke tests — one per quantization family — plus a throughput benchmark and the ternary downloader.

| Script | Target model | Prerequisite | Purpose |
|--------|--------------|--------------|---------|
| `scripts/cli.sh [metal\|cuda]`                        | `models/Bonsai-8B.gguf`            | `oxibonsai pull bonsai-8b` (see [Quick Start](#quick-start))       | Build + end-to-end CLI test on **1-bit Bonsai-8B** |
| `scripts/cli_ternary.sh [metal\|cuda\|cuda-scirs]`    | `models/Ternary-Bonsai-1.7B.gguf` (default; `--model` to override) | **run `scripts/download_ternary.sh` first** | Build + end-to-end CLI test on **Ternary Bonsai** with a tok/s summary line |
| `scripts/bench_ternary.sh`                            | `models/Ternary-Bonsai-1.7B.gguf`  | `scripts/download_ternary.sh`                        | CPU vs Metal throughput benchmark (averaged over N runs) |
| `scripts/download_ternary.sh [8b\|4b\|1.7b]`          | —                                  | `pip install huggingface_hub`                        | Download Ternary Bonsai safetensors from HF and convert to GGUF |

Each CLI script:
1. Builds a `--release` binary with the requested feature flags
2. Runs inference (`oxibonsai run`)
3. Prints model info (`oxibonsai info`) and validates the GGUF (`oxibonsai validate`)
4. Reports the measured tok/s

```bash
# 1-bit flow (Bonsai-8B)
./scripts/cli.sh                 # CPU SIMD
./scripts/cli.sh metal           # Metal GPU (macOS)
./scripts/cli.sh cuda            # CUDA GPU  (Linux/Windows; validated on Linux x86_64, RTX A4000)

# Ternary flow — fetch + convert once, then run as many times as you like
./scripts/download_ternary.sh 1.7b
./scripts/cli_ternary.sh         # CPU SIMD
./scripts/cli_ternary.sh metal   # Metal GPU — fused TQ2 full-forward path
./scripts/cli_ternary.sh cuda    # native CUDA backend
./scripts/bench_ternary.sh       # CPU vs Metal, 3-run average + best
```

## Measured Throughput

Rows marked CUDA are from one NVIDIA RTX A4000 (Ampere, compute capability 8.6, default 140 W clocks, CUDA 12.0, driver 550.144, Xeon Gold 5315Y host, quiet box, id-42 ternary files), measured on 2026-10-07; every other row is from an Apple M3 with 24 GB. The M3 host was under background load (load average 6–22) during most of these runs, so they are ranges, not single best figures, and they move with load. "Fused full-forward" = single GPU command buffer per token.

| Model | Backend | What | Result |
|-------|---------|------|-------:|
| Ternary-Bonsai-1.7B | **Metal** (fused TQ2) | decode, single stream | ~50 tok/s (50–57 across runs) |
| Ternary-Bonsai-1.7B | CPU SIMD (NEON, f32) | decode | ~7–8 tok/s |
| Ternary-Bonsai-1.7B | CPU, opt-in `neon-i8mm` | decode through the engine | 29–36 tok/s (8.76× the f32 CPU tier) |
| Bonsai-8B           | Metal (fused Q1) | decode | ~14.6 tok/s |
| **Bonsai 2 27B** (PQ2\_0 / PTQ1\_0) | **Metal** hybrid runner | decode | 6.9–8.4 tok/s |
| Bonsai 2 27B (PQ2\_0 / PTQ1\_0) | CPU | decode | 1.0–1.3 tok/s |
| Bonsai 2 27B | Metal hybrid runner | batched prefill at 512 tokens | 4.1–6.3× its own decode rate |
| Ternary-Bonsai-1.7B | Metal | fused prefill vs sequential decode, 256 tokens | 11.9–12.5× faster per token (4096 tokens: 7.4–7.6 s vs 130–134 s) |
| Bonsai-8B           | Metal | fused prefill vs sequential decode, 256 tokens | 8.05–8.2× faster per token (4096 tokens: 27.5–29.4 s vs 304–322 s) |
| Dense models | Metal | 2000-token embedding | 3.3 s (1.7B), 13 s (8B models) |
| Bonsai 2 27B | Metal | 256 × 192 image prompt end to end | 14 s (78 s on the CPU path) |
| Ternary-Bonsai-1.7B | **CUDA** (fused TQ2), RTX A4000 | greedy decode, 256 tokens after a 16-token warm-up (`oxibonsai benchmark`) | 44.4 tok/s |
| Ternary-Bonsai-1.7B | CUDA, RTX A4000 | `scripts/cli_ternary.sh cuda` (sampled, temperature 0.7 / top-p 0.9, 100 tokens, prefill included) | 39.6 tok/s (an earlier release: ~21.9 tok/s, measured the same way on other NVIDIA hardware) |
| Ternary-Bonsai-8B   | CUDA (fused TQ2), RTX A4000 | greedy decode, 256 tokens after a 16-token warm-up (`oxibonsai benchmark`) | 10.8 tok/s |
| Bonsai-8B           | CUDA (fused Q1), RTX A4000 | greedy decode, 256 tokens after a 16-token warm-up (`oxibonsai benchmark`) | 20.6 tok/s |

The 27B decode bar the release was held to is 5 tok/s, and both formats cleared it. The Bonsai-8B 4096-token "before" figure was measured on a 512-token prompt because the pre-fix 4096-token arm did not finish within 900 s. Numbers come from `scripts/bench_ternary.sh`, `scripts/cli_ternary.sh` and the release-gate legs; the CUDA greedy rows from `oxibonsai benchmark --tokens 256 --warmup 16 --temperature 0 --seed 42`. Q4_0 / Q8_0 / K-quant / FP8 decode on CUDA is PCIe-bound and can be slower than the CPU (see [Known Limitations](#known-limitations)). Engine-pool replicas do **not** multiply GPU throughput: 8 concurrent requests on the real 1.7B served 52.2 / 54.2 / 54.5 tok/s aggregate with pools of 1 / 2 / 3 replicas (weights stay 457 MB resident for all three), so replicas buy isolation and latency fairness, not throughput.

Sampled decode (the default `temperature 0.7`, `top_k 40`) reads the full logit row on the host, which costs a little more per token than greedy's 4-byte GPU argmax readback: on the real Ternary-Bonsai-1.7B, in one process (release build, load average about 12), 62.2 tok/s greedy against 53.1 tok/s default-sampled (0.85x). `real_model_default_sampled_decode_keeps_pace_with_greedy` repeats that measurement and requires at least 0.6x. A GPU top-k candidate route exists (`InferenceEngine::set_sampled_topk(SampledTopKConfig::gpu_candidates())`, a library opt-in with byte-identical output) but is off by default, because its selection kernel costs about 25 ms per token at a 151 669-token vocabulary: 22.2 tok/s (0.33x) in the same measurement.

## Configuration

OxiBonsai supports TOML configuration files with `--config`:

```toml
[model]
model_path = "models/Ternary-Bonsai-1.7B.gguf"
max_seq_len = 4096
# rope_scaling = "auto"        # auto | on | off

[sampling]
temperature = 0.7
top_k = 40
top_p = 0.9
repetition_penalty = 1.0
# seed = 42

[server]
host = "127.0.0.1"
port = 8080

[observability]
log_level = "info"
json_logs = false
```

An unreadable file or an unknown key is a hard error naming the path. See [`docs/CLI.md`](docs/CLI.md) for every key and the precedence (`--flag` > `--config` file > shell env var > `.env` > built-in default).

## Crate Structure

```
oxibonsai/
├── crates/
│   ├── oxibonsai-core/        GGUF reader/writer, quantization block types (Q1/TQ2,
│   │                          K-quants, PQ2_0/PTQ1_0/Q2_0), model + hybrid + Hadamard config
│   ├── oxibonsai-kernels/     Dequant / GEMV / GEMM kernels (SIMD tiers, INT8 tier,
│   │                          tiled, parallel), Gated-DeltaNet, FWHT, M-RoPE + GPU backends:
│   │                            gpu_backend/metal_*       (Metal graph, fused full-forward,
│   │                                                       hybrid runner, vision tower)
│   │                            gpu_backend/cuda_*        (native NVRTC kernels; validated on one RTX A4000)
│   │                            gpu_backend/scirs2_backend (scirs2-core CUDA/Metal)
│   ├── oxibonsai-tokenizer/   Pure Rust BPE tokenizer, GGUF vocabulary, Jinja chat templates
│   ├── oxibonsai-model/       Dense Qwen3 + hybrid Qwen3.5 forward passes (GQA, SwiGLU, RoPE,
│   │                          RMSNorm, KV cache), weight loaders, converters, ViT vision tower
│   ├── oxibonsai-rag/         RAG pipeline (chunking, embedders, vector store)
│   ├── oxibonsai-runtime/     Inference engine, sampling, OpenAI-compatible server,
│   │                          SSE streaming, metrics, health, reasoning + tool calls
│   ├── oxibonsai-eval/        Evaluation harness (ROUGE, BLEU, chrF, METEOR, perplexity, MMLU, ...)
│   ├── oxibonsai-serve/       Standalone server binary
│   ├── oxibonsai-image/       Bonsai-Image: text encoder, FLUX.2 DiT, VAE, PNG (Metal/CUDA/CPU)
│   ├── oxibonsai-testkit/     Dev-only: GGUF fixtures, parity and capability helpers
│   └── oxibonsai/             Facade crate (feature-gated re-exports)
├── src/main.rs                CLI entry point
├── src/cli/                   Subcommands: run, chat, serve, info, build-info, benchmark,
│   │                          convert, quantize, validate, eval, tokenizer, pull, image, repl
│   ├── repl.rs                `oxibonsai repl` — resident `ImageSession` (loads
│   │                          DiT/VAE/TE once, renders many prompts); Kitty
│   │                          graphics protocol inline display (Ghostty detection)
│   └── term.rs                Terminal detection helpers (Ghostty / Kitty protocol)
├── benches/                   Criterion kernel benchmarks
├── examples/                  Usage examples (plus per-crate examples/ directories)
├── docs/                      Model matrix, CLI reference, deployment guide, imagen guide
├── tests/                     Integration + feature flag tests
└── scripts/                   ci.sh / release-gate.sh / publish.sh, CLI smoke tests, downloaders
```

## Examples

See the `examples/` directory (and the per-crate `examples/` directories):

- `basic_inference.rs` — Load a model and run single-shot inference
- `streaming.rs` — Server-sent event streaming
- `custom_sampling.rs` — Custom sampling parameters and presets
- `crates/oxibonsai-rag/examples/rag_qa.rs` — Retrieval-augmented question answering, no model needed
- `crates/oxibonsai-eval/examples/eval_mmlu.rs` — MMLU-style multiple-choice evaluation, end to end
- `crates/oxibonsai-image/examples/text_to_image.rs` — Bonsai-Image text-to-image through the library API

```bash
# 1-bit
cargo run --example basic_inference -- --model models/Bonsai-8B.gguf

# Ternary
cargo run --example basic_inference -- --model models/Ternary-Bonsai-1.7B.gguf

# RAG and eval examples (no model file needed)
cargo run -p oxibonsai-rag --example rag_qa
cargo run -p oxibonsai-eval --example eval_mmlu
```

## COOLJAPAN Ecosystem

```
OxiBonsai (Pure Rust sub-2-bit LLM inference — CPU, Metal, CUDA)
  ├── SciRS2   (scirs2-core 0.6.x — GPU abstraction tier, SIMD helpers)
  ├── OxiArc   (oxiarc-deflate 0.4.x — DEFLATE for the PNG encoder/decoder)
  ├── OxiHTTP  (oxihttp 0.2.x, with the Pure-Rust OxiTLS stack — downloads)
  └── OxiONNX  (oxionnx-proto — ONNX protobuf for `convert --onnx`)
```

### Pure Rust guarantee

All **default-feature** dependencies are Pure Rust — zero C/C++/Fortran, zero FFI. GPU backends (`metal`, `native-cuda`, `cuda`) are opt-in features that bring in vendor drivers. The guarantee is enforced, not just stated:

- `scripts/check_pure_rust.sh` (a `scripts/ci.sh` stage) resolves exactly the default dependency set that `cargo build --workspace` pulls in and fails if `cc`, `cmake` or `bindgen` is reachable from it, printing the reverse-dependency path; it also heuristically scans the `build.rs` of every default-resolved dependency for a compiler shell-out.
- `deny.toml` (checked by `cargo deny check bans licenses sources advisories`) bans the COOLJAPAN-superseded and C-backed crates (openblas, bincode, rustfft, z3, rusqlite, zip/flate2/zstd/bzip2/lz4/tar/snap/brotli/miniz\_oxide, `aws-lc-sys`, `openssl-sys`, ...) and restricts licences to a permissive allow-list.
- `core-foundation-sys` (reached through `chrono` → `iana-time-zone`) is the only `-sys` node in the default dependency graph. It is the platform binding to Apple's CoreFoundation framework — no C/C++/Fortran is compiled — and is allowed.
- The tokenizer uses the Pure-Rust native backend by default; the Hugging Face `tokenizers` crate is opt-in (`hf-tokenizer`), and its regex engine is the Pure-Rust `fancy-regex`. The native tokenizer's per-pretokenizer `Split` regex also uses `fancy-regex` (pure Rust, workspace-managed, backtrack-capped) rather than a hand-written splitter, and its `Llama3` pre-tokenizer pattern is an approximation of the GPT-2 one.
- The dev-only `criterion` → `alloca` → `cc` edge is a recorded exception: Cargo builds every dev-dependency whenever any test target is selected, so `cargo test --no-run` still compiles `cc`. It never reaches a default `cargo build` or an installed binary.
- The minimum supported Rust version is **1.89** (the dependency graph does not resolve below it).

## Development Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 0 | Foundation (workspace, GGUF loader, metadata) | ✅ |
| Phase 1 | 1-Bit Kernels (dequant, GEMV, GEMM) | ✅ |
| Phase 2 | Transformer Engine (Qwen3-8B forward pass) | ✅ |
| Phase 3 | Inference Runtime (KV cache, sampling, CLI) | ✅ |
| Phase 4 | Production Hardening (SIMD, parallel, tests, observability) | ✅ |
| Phase 5 | Ecosystem Integration (SSE streaming, WASM, API, Bonsai family) | ✅ |
| Phase 6 | Advanced Infrastructure (Multi-GPU, CUDA/Metal, PagedAttention) | ✅ * |
| Phase 7 | Production Features (model merging, flash decoding, RAG, eval) | ✅ |
| Phase 8 | Final Polish (K-quant, streaming GGUF, kernel tuning, tests) | ✅ |
| Phase 9 | Ternary Bonsai (TQ2\_0\_g128 kernels, model variants, GGUF surface, export) | ✅ |
| Phase 10 | Ternary CPU SIMD tiers (AVX2 / AVX-512 / NEON TQ2 GEMV) | ✅ |
| Phase 11 | Metal TQ2 GEMV + per-kernel dispatch | ✅ |
| Phase 12 | Native CUDA backend (NVRTC, fused Q1 + TQ2 full-forward) | ✅ † |
| Phase 13.x | Fused Metal TQ2 full-forward (single command buffer, ~13× speedup on 1.7B) | ✅ |
| Phase 13.y | Ternary LM head on GPU — closes all 7 `OutputWeight::Ternary` guard sites (4 Metal + 3 CUDA); +5 tok/s on Metal | ✅ |
| 0.2.4 | **Bonsai 2 27B**: `qwen35` hybrid (Gated DeltaNet), PQ2\_0 / PTQ1\_0 / Q2\_0 g64, Hadamard contract, Metal hybrid runner, chat template + reasoning + tools, vision (CPU and Metal), hybrid embeddings, opt-in INT8 tier, dense Metal prefill fix | ✅ |
| 0.2.4 | Production audit: K-quant layouts byte-exact against ggml, safe server defaults, verified downloads, `scripts/ci.sh` + `scripts/release-gate.sh`, Pure-Rust policy gate, rustdoc `-D warnings` | ✅ |

\* Phase 6 items marked complete are implemented and tested, but several are standalone building blocks not yet wired into the default inference path, or are honestly-labeled simulations rather than the literal capability their name suggests — see [Known Limitations](#known-limitations) immediately below for the itemized, current-reality breakdown.

† Phase 12 and the 0.2.4 CUDA work were run on one RTX A4000 (CUDA 12.0) for this release, except the Bonsai 2 hybrid kernels; [Known Limitations](#known-limitations) lists what was not run.

## Known Limitations

Everything below is implemented, tested, and either self-documented in its own module doc comments or tracked in [`TODO.md`](TODO.md) — nothing here is a secret, but several roadmap/README lines elsewhere in this document are easy to over-read as "fully wired into inference." This section is the honest, consolidated version.

**Acceleration & scaling**

- **CUDA is validated on one GPU only.** On one NVIDIA RTX A4000 (Ampere, compute capability 8.6), CUDA 12.0, driver 550.144, x86_64 Ubuntu 22.04 (2026-10-07): the kernel sources compiled under `nvcc` (31/31, before the `gemv_q4_0_pf` fix); 168 kernel tests pass; 48-token greedy output is byte-identical to the CPU for Bonsai-8B, Ternary-Bonsai-1.7B/8B (id-42 files), and Q4_0 / Q8_0 / six K-quant / two FP8 fixtures derived from Ternary-Bonsai-1.7B with a quantized LM head; checklist items CUDA-P11, P14, P15 and P18 and the keyed graph slot pass on real weights; the image pipeline's CUDA arm reproduced a prior render at PSNR 76–78 dB (not a repository golden). **Not run:** the Bonsai 2 27B files and hybrid kernels (CUDA-P01–P10; there is no CUDA `qwen35` forward), the fused gate‖up GEMV in the sliding-window and stats forwards (P16/P17), Turing/Pascal GPUs, aarch64 Linux, Windows and multi-GPU hosts. The full `scripts/ci.sh --release` and `release-gate.sh --require-cuda` runs on the CUDA host are pending, and the macOS release gate still checks CUDA kernel syntax approximately (no `nvcc` there; an explicit waiver). The remaining open checklist items are in `TODO.md`. (The earlier "cap-of-8" batch-column bug is fixed: the CUDA prefill kernels chunk correctly.)
- **CUDA limits measured on the A4000.** Q4_0 / Q8_0 / K-quant / FP8 decode uploads each weight matrix on every GEMV (correct but PCIe-bound; a Q8_0 fixture decodes at 3.8–4.0 tok/s on CUDA against 6.2 on the AVX-512 CPU); the Q1 batch prefill is 0.6–0.8x the per-token path; K-quant / FP8 warm-up takes 3–10 s; Ternary-Bonsai-8B holds 5739 MiB of VRAM for a 2081 MiB file. `oxibonsai quantize` keeps the LM head in F32, so its output never takes the CUDA quantized-LM-head paths. A library engine built with a CPU-tier dispatcher still runs FP8 linears on the GPU (`--backend cpu` is unaffected). With a Q1/ternary model the batch prefill keeps the K/V on the device and the per-token path (≤ 16 tokens) reads the host cache, so a following 2–16-token window is refused (`GPU_FALLBACK_REQUIRES_CACHE_REBUILD`); both chunk planners (the engine's, behind the server's 512-token windows, and the model's, behind `--prefill-chunk N` and the default 4096-token plan) fold such a tail into the previous window when the chunk exceeds 16 tokens, so only `--prefill-chunk 2`–`16` or a library caller driving `forward_prefill` with its own windows can hit it (fold verified on the CPU; the CUDA re-run is pending). Run one Q1 model per process: the Q1 weight-cache fingerprint hashes host addresses only. Details in the [changelog](CHANGELOG.md).
- **`cuda` (scirs2-core) tier is a CPU fallback, not GPU acceleration.** `is_accelerated()` always returns `false` for it because scirs2-core retired its cudarc-based backend in 0.6.x; the dispatcher silently falls back to CPU SIMD. Output is correct — there is no garbage-token risk — it is simply not GPU-accelerated. Use `native-cuda` for NVIDIA throughput.
- **"True multi-GPU inference" is a rayon CPU simulation of NCCL-style collectives**, not real inter-GPU communication. `multi_gpu.rs`'s types are named `SimulatedDeviceMesh` / `SimulatedCollectives` for that reason, and are not wired into any actual multi-device dispatch path.
- **"Distributed serving" is an in-memory routing topology, not a networked serving layer.** `distributed.rs` implements consistent-hash-ring request routing and a node registry entirely in-process; its own doc comment states "no actual TCP connections are made."
- **GPU concurrency.** Each Metal engine replica owns its own session, but replicas multiply isolation, not throughput (measured +4–6 % aggregate for 2–3 replicas). `OXIBONSAI_METAL_MAX_SESSIONS` (default 4) is reported by `oxibonsai info` but **not enforced** when a session opens (a Metal vision tower is a second session; each pool replica adds one). The CUDA tier keeps its process-global graph, so the pool clamps to one replica there.

**GPU platform coverage**

- **Metal has full Q4_0/Q8_0/K-quant (Q2_K–Q8_K) and FP8 batch-prefill coverage.** Metal GEMV kernels + host dispatch exist for all 8 standard/K-quant formats, and the model-layer `Linear*::forward()` implementations try Metal first before falling back to CPU on any error. Metal FP8 batch prefill is a hybrid design: the heavy linear projections run as batched FP8 GEMMs on the GPU, while q/k-norm, RoPE, attention, and the K/V store stay on the CPU against the same cache per-token decode reads (so it avoids the split-KV-cache bug class by construction). The K-quant GEMV kernels (Metal and CUDA) were *wrong* before 0.2.4 and are now ggml-exact; see the [changelog](CHANGELOG.md).
- **CUDA batch prefill coverage.** Q1, ternary, Q4_0 and Q8_0 prompts of more than 16 tokens at position 0 take the CUDA batch prefill; for Q4_0/Q8_0 it reads the prompt K/V back into the host cache that per-token decode reads (validated as CUDA-P11 on an RTX A4000). Q4_0/Q8_0 prefills at position > 0, and every K-quant or FP8 prefill, take the bit-correct sequential per-token CUDA path: the K-quant and FP8 batch-prefill entry points write a GPU-private KV cache with no read-back (the KV-cache-handoff bug class fixed 2026-07-20), so they refuse by construction, and a context-length guard turns an over-long FP8 prompt into a clean `SequenceTooLong` instead of a panic. **Practical impact: none for correctness**; the cost is that K-quant and FP8 CUDA prompt prefill runs the sequential per-token path. `OXIBONSAI_FORCE_CUDA_SPLIT_PREFILL` is no longer honoured: setting it logs one error and changes nothing. Tracked in `TODO.md`.

**Building blocks implemented and tested, but experimental or not wired into the default forward path**

`BonsaiModel::forward`/`forward_prefill` run a dense GQA-attention + dense-SwiGLU-FFN Qwen3 layer stack (or the hybrid `qwen35` stack). Sliding-window attention is wired for models that declare `<arch>.attention.sliding_window` (CPU path; the fused GPU routes decline such models), and YaRN / rope scaling is applied from the GGUF's own `rope.scaling.*` keys. The following are complete, unit-tested library primitives usable by external consumers of the crates, but none of them execute inside that default forward path today:

- **PagedAttention** (`PagedKvCache`/`BlockPool`/`BlockTable`/`KvPage`) — experimental; the shipping engine uses the contiguous `KvCache`.
- **KV-cache quantization** (`QuantizedKvCache`, `Fp8KvCache`, `kv_cache_quant.rs`) — never constructed by the model or runtime. `oxibonsai-runtime`'s `KvCachePolicy` computes an adaptive FP16/Q8/Q4 *target level* from memory pressure, but only feeds a Prometheus gauge and an admin endpoint — it does not reconfigure the actual cache (hybrid models can opt into a sparse `f16` KV cache chosen once at load time).
- **Attention-variant layers** — attention sink / StreamingLLM, sparse attention (local/strided/BigBird/Longformer/dilated), flash decoding's alternate path, cross-attention, Mixture-of-Depths, and the MoE router/expert modules are standalone primitives with no call sites in `TransformerBlock::forward` outside their own test suites.
- **Library-only runtime features** — speculative decoding (`SpeculativeDecoder::generate_verified`, which refuses hybrid models), the prefix cache (off for hybrid models), continuous batching, the request queue and the circuit breaker are exported library APIs, not wired into `oxibonsai serve`.

**Hybrid (`qwen35`) models**

- Prefix caching, speculative decoding and any other KV-cursor rewind are refused for hybrid models (a Gated-DeltaNet state cannot be rolled back by moving a cursor).
- Deferred design items: chunked UT-transform Gated-DeltaNet prefill, a recurrent-state ring buffer for prefix caching, the MTP/NextN draft head, Hadamard KV-cache rotation, video input, and the `qwen35moe` / `qwen3next` variants — see `TODO.md`.
- The CPU vision tower takes 23–30 s for a 768 × 768 image; the Metal tower (2.6–2.7 s) is the fast path. Remote `http(s)` image URLs are fetched only with `--allow-image-url-fetch`, from public addresses (and allowlisted hosts) only.
- Metal waits on the 27B hybrid runner and the Metal vision tower are unbounded: a stalled GPU holds the replica (and the tower) while the client receives its `504`, and a Metal engine has no CPU-tower fallback.

**Other**

- **Real-model coverage.** The real-model gates cover the two 27B variants on the reference host (PQ2\_0, PTQ1\_0); the other four 27B language files are covered by header, loader and fixture tests only. The eval logit tasks have never been run against a real model, and `avx512-vnni` has never run on VNNI hardware.
- **Retrieval.** The RAG vector store is a brute-force `entries × dim` scan (50,000 entries of 384 dimensions is about 73 MiB and 19 million multiply-adds per query); an HNSW index behind an `ann` feature is a deliberately deferred, project-sized item.
- **Speculative decoding** — the production two-engine path is `SpeculativeDecoder::generate_verified(&mut target_engine, ...)`: it drafts against the delta-KV path and verifies against a real, separate target `InferenceEngine`, and its accepted output is token-identical to plain greedy decoding of the target model. The originally-shipped `generate_speculative`/`verify` API is retained only as a `#[doc(hidden)]` synthetic test harness.

## Sponsorship

OxiBonsai is developed and maintained by **COOLJAPAN OU (Team Kitasan)**.

The COOLJAPAN Ecosystem represents one of the largest Pure Rust scientific computing efforts in existence — spanning 40+ projects, 500+ crates, and millions of lines of Rust code across scientific computing, machine learning, quantum computing, geospatial analysis, legal technology, multimedia processing, and more. Every line is written and maintained by a small dedicated team committed to a C/Fortran-free future for scientific software.

If you find OxiBonsai or any COOLJAPAN project useful, please consider sponsoring to support continued development.

[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-red?logo=github)](https://github.com/sponsors/cool-japan)

**[https://github.com/sponsors/cool-japan](https://github.com/sponsors/cool-japan)**

Your sponsorship helps us:
- Maintain and expand the COOLJAPAN ecosystem (40+ projects, 500+ crates)
- Keep the entire stack 100% Pure Rust — no C/Fortran/system library dependencies
- Develop production-grade alternatives to OpenCV, FFmpeg, SciPy, NumPy, scikit-learn, PyTorch, TensorFlow, GDAL, and more
- Provide long-term support, security updates, and documentation
- Fund research into novel Rust-native algorithms and optimizations

## License

Apache License, Version 2.0

Copyright 2026 COOLJAPAN OU (Team KitaSan)
