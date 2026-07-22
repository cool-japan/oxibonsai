# OxiBonsai CLI Reference

This document is the complete command-line reference for the two OxiBonsai
binaries:

- **`oxibonsai`** — the main inference engine (text generation, image
  generation, chat, server, model tooling). Argument parsing is `clap`-based,
  so every subcommand supports `--help`.
- **`oxibonsai-serve`** — a standalone OpenAI-compatible HTTP server with its
  own hand-rolled argument parser and a layered TOML/env/CLI configuration
  system.

All flags and defaults below are taken directly from the source
(`src/main.rs` and `crates/oxibonsai-serve/src/{args.rs,env.rs,config.rs}`).

## Contents

- [`oxibonsai` — global usage](#oxibonsai--global-usage)
  - [`run`](#run)
  - [`image`](#image)
  - [`repl`](#repl)
  - [`chat`](#chat)
  - [`serve`](#serve)
  - [`info`](#info)
  - [`benchmark`](#benchmark)
  - [`quantize`](#quantize)
  - [`validate`](#validate)
  - [`convert`](#convert)
  - [`eval`](#eval)
  - [`tokenizer`](#tokenizer)
- [`oxibonsai-serve` — standalone server](#oxibonsai-serve--standalone-server)
- [Environment variables](#environment-variables)
  - [`OXI_*` (oxibonsai binary)](#oxi_-oxibonsai-binary)
  - [`OXIBONSAI_*` (oxibonsai-serve server)](#oxibonsai_-oxibonsai-serve-server)
  - [Precedence](#precedence)

---

## `oxibonsai` — global usage

```text
oxibonsai [--config <PATH>] <COMMAND> [OPTIONS]
```

| Flag | Type | Default | Help |
|------|------|---------|------|
| `--config <PATH>` | string | — | Path to an OxiBonsai TOML configuration file. Global; applies to every subcommand. |

`--version` and `--help` are available globally and on every subcommand.

The `oxibonsai` binary auto-loads a `.env` file from the current directory (or
any parent) at startup. This does **not** override real environment variables,
so precedence stays `--flag` > shell env > `.env` > built-in default. See
[Environment variables](#environment-variables).

---

### `run`

Run inference on a GGUF model and stream generated text to stdout.

| Flag | Short | Type | Default | Help |
|------|-------|------|---------|------|
| `--model <PATH>` | `-m` | string | env `OXI_MODEL` | Path to the GGUF model file. Required unless `OXI_MODEL` is set. |
| `--prompt <TEXT>` | `-p` | string | *(required)* | Prompt text. Use `-` to read the prompt from stdin. |
| `--max-tokens <N>` | | usize | `256` | Maximum number of tokens to generate. |
| `--temperature <F>` | | f32 | `0.7` | Sampling temperature (`0.0` = greedy). On macOS+Metal, `0.0` enables the greedy GPU argmax path. |
| `--top-k <N>` | | usize | `40` | Top-k sampling (`0` = disabled). |
| `--top-p <F>` | | f32 | `0.9` | Top-p (nucleus) sampling. |
| `--seed <N>` | | u64 | `42` | Random seed. |
| `--max-seq-len <N>` | | usize | `4096` | Maximum sequence length (prompt + generated). |
| `--tokenizer <PATH>` | | string | env `OXI_TOKENIZER`, else auto-detect | Path to `tokenizer.json`. If omitted, OxiBonsai auto-detects it next to the model (and a few conventional sibling/parent locations). |

The internal repetition penalty is fixed at `1.1` for this command.

```bash
oxibonsai run \
  --model models/Ternary-Bonsai-1.7B.gguf \
  --prompt "Explain ternary quantization in one sentence." \
  --max-tokens 256 --temperature 0.7 --top-k 40 --top-p 0.9
```

---

### `image`

Generate an image from a text prompt using Bonsai-Image (FLUX.2-Klein):
text-encoder → DiT → VAE → PNG. The pipeline runs end-to-end in pure Rust.

| Flag | Short | Type | Default | Help |
|------|-------|------|---------|------|
| `--prompt <TEXT>` | `-p` | string | *(required)* | Text prompt. Use `-` to read from stdin. |
| `--out <PATH>` | `-o` | string | *(required)* | Output PNG path. |
| `--seed <N>` | | u64 | `42` | RNG seed for the initial noise. |
| `--steps <N>` | | usize | `4` | Number of Euler (flow-matching) sampler steps. |
| `--width <N>` | | usize | `512` | Image width in pixels. |
| `--height <N>` | | usize | `512` | Image height in pixels. |
| `--dit <PATH>` | | string | env `OXI_DIT_GGUF`, else `/tmp/parity.gguf` | DiT (FLUX.2-Klein ternary) GGUF path. |
| `--vae <PATH>` | | string | env `OXI_VAE_WEIGHTS`, else `/tmp/bonsai_golden/vae/weights` | VAE decoder weights. Accepts either a `.safetensors` file or a directory of `.npy` tensors (see note below). |
| `--te <PATH>` | | string | env `OXI_TE_4BIT`, else env `OXI_TE_WEIGHTS`, else `/tmp/bonsai_golden/te/weights` | Text-encoder weights: a 4-bit MLX `model.safetensors` file or an f32 `.npy` directory. A path ending in `.safetensors` selects the 4-bit MLX loader; anything else is treated as an `.npy` directory. |
| `--tokenizer <PATH>` | | string | env `OXI_TE_TOKENIZER_DIR`, else the TE dir | Tokenizer directory containing `tokenizer.json`. When `--te` is a `.safetensors` file, this defaults to that file's parent directory. |

> **VAE weights — file or directory.** `--vae` / `OXI_VAE_WEIGHTS` accepts
> **either** a single `.safetensors` file **or** a directory of per-tensor
> `.npy` exports (`decoder.*.npy`, `post_quant_conv.*.npy`, …). The loader
> picks the right reader from the path.

```bash
oxibonsai image \
  --prompt "a tiny bonsai tree in a ceramic pot" \
  --out bonsai.png \
  --seed 42 --steps 4 --width 512 --height 512
```

With a populated `.env` (`OXI_DIT_GGUF`, `OXI_VAE_WEIGHTS`, `OXI_TE_4BIT`,
`OXI_TE_TOKENIZER_DIR`) the path flags can be omitted:

```bash
oxibonsai image --prompt "a tiny bonsai tree in a ceramic pot" --out bonsai.png
```

---

### `repl`

Interactive image generation. Loads the DiT, VAE, and text encoder **once** into
a resident session, then renders prompts you type without re-loading or
re-dequantising weights. The text encoder is held resident (its dequantised f32
weights, ~16 GB, stay cached across prompts), so this is meant for high-memory
machines — it trades RAM for not re-paying the per-prompt warm-up. On a
kitty-graphics terminal (Ghostty) each image is shown **inline**; elsewhere it is
written to a file (and optionally opened in a viewer).

Model paths resolve exactly like [`image`](#image) (flag → env → default).

| Flag | Type | Default | Help |
|------|------|---------|------|
| `--seed <N>` | u64 | `42` | Initial RNG seed (change at runtime with `:seed`). |
| `--steps <N>` | usize | `4` | Initial sampler steps (`:steps` / `:fast` / `:hq`). |
| `--width <N>` | usize | `512` | Initial width in pixels. |
| `--height <N>` | usize | `512` | Initial height in pixels. |
| `--cpu-te` | flag | off | Run the text-encoder GEMM on the CPU instead of the Metal GPU. |
| `--dit <PATH>` | string | env `OXI_DIT_GGUF`, else `/tmp/parity.gguf` | DiT GGUF path. |
| `--vae <PATH>` | string | env `OXI_VAE_WEIGHTS` | VAE weights (file or `.npy` dir). |
| `--te <PATH>` | string | env `OXI_TE_4BIT`, else env `OXI_TE_WEIGHTS` | Text-encoder weights. |
| `--tokenizer <PATH>` | string | env `OXI_TE_TOKENIZER_DIR`, else the TE dir | Tokenizer directory. |

Inside the REPL, a bare line is a prompt; `:`-prefixed lines are commands:

| Command | Effect |
|---------|--------|
| `:fast` | Preset: 2 steps, 384×384 (snappy preview). |
| `:hq` | Preset: 8 steps, 512×512 (higher quality). |
| `:steps N` / `:seed N` | Set a single parameter. |
| `:size WxH` | Set output size (or `:size N` for square). |
| `:out PATH` | Write to `PATH` (no arg → auto `oxibonsai-repl-NNN.png`). |
| `:open on\|off` | Open each image in a viewer (non-inline terminals). |
| `:show` | Print current settings. |
| `:help` | Command reference. |
| `:quit` | Exit (also Ctrl-D). |

```bash
# With OXI_DIT_GGUF / OXI_VAE_WEIGHTS / OXI_TE_4BIT set in the environment:
oxibonsai repl
oxibonsai> :fast
oxibonsai> a red fox in the snow
oxibonsai> :hq
oxibonsai> a red fox in the snow
```

After loading, each render reuses the resident weights, so per-prompt time is
just compute (text-encode + DiT sampling + VAE decode); the one-time load and
encoder warm-up are paid once at startup.

---

### `chat`

Interactive multi-turn conversation. Type `quit` / `exit` or press Ctrl-D to
leave; `/reset` clears the conversation context. Ctrl-C interrupts the current
generation without exiting.

| Flag | Short | Type | Default | Help |
|------|-------|------|---------|------|
| `--model <PATH>` | `-m` | string | env `OXI_MODEL` | Path to the GGUF model file. Required unless `OXI_MODEL` is set. |
| `--max-tokens <N>` | | usize | `512` | Maximum number of tokens to generate per turn. |
| `--temperature <F>` | | f32 | `0.7` | Sampling temperature (`0.0` = greedy). |
| `--top-k <N>` | | usize | `40` | Top-k sampling (`0` = disabled). |
| `--top-p <F>` | | f32 | `0.9` | Top-p (nucleus) sampling. |
| `--seed <N>` | | u64 | `42` | Random seed. |
| `--max-seq-len <N>` | | usize | `4096` | Maximum sequence length. |
| `--tokenizer <PATH>` | | string | env `OXI_TOKENIZER`, else auto-detect | Path to `tokenizer.json`. |

The internal repetition penalty is fixed at `1.1` for this command.

```bash
oxibonsai chat --model models/Bonsai-8B.gguf --temperature 0.7
```

---

### `serve`

Start an OpenAI-compatible API server backed by the in-process engine. This
subcommand is gated behind the `server` build feature.

| Flag | Short | Type | Default | Help |
|------|-------|------|---------|------|
| `--model <PATH>` | `-m` | string | env `OXI_MODEL` | Path to the GGUF model file. Required unless `OXI_MODEL` is set. |
| `--host <HOST>` | | string | `127.0.0.1` | Host to bind to. |
| `--port <PORT>` | | u16 | `8080` | Port to listen on. |
| `--max-seq-len <N>` | | usize | `4096` | Maximum sequence length. |
| `--tokenizer <PATH>` | | string | env `OXI_TOKENIZER`, else auto-detect | Path to `tokenizer.json`. |
| `--pool-size <N>` | | usize | env `OXIBONSAI_ENGINE_POOL_SIZE`, else `min(4, CPU cores)` | Number of engine replicas. Auto-clamped to `1` on the GPU/Metal tier. Replicas share one token-embedding table, so each extra replica only costs a KV cache. |
| `--bearer-token <TOK>` | | string | env `OXIBONSAI_BEARER_TOKEN` | Require this bearer token (constant-time comparison) on every endpoint except `/health` and `/metrics` — this also gates the mutating `/admin/*` endpoints. When unset, the server logs an explicit warning and runs unauthenticated; only safe behind `--host 127.0.0.1` or another trusted network boundary. |
| `--max-concurrent-requests <N>` | | usize | `32` | Maximum number of requests admitted concurrently; requests beyond this bound are rejected with `503` instead of queuing unbounded. |
| `--request-timeout-ms <N>` | | u64 | `60000` | Per-request timeout in milliseconds before a request is aborted with `408`. |
| `--rag` | | bool | `false` | Also mount the RAG HTTP API (`/rag/index`, `/rag/query`, `/rag/stats`) alongside the OpenAI-compatible endpoints. Requires building with the `rag` Cargo feature. |

```bash
oxibonsai serve \
  --model models/Ternary-Bonsai-1.7B.gguf \
  --host 127.0.0.1 --port 8080

# With bearer-token auth + admission limits (production hardening):
oxibonsai serve --model models/Ternary-Bonsai-1.7B.gguf \
  --bearer-token "$(openssl rand -hex 32)" \
  --max-concurrent-requests 32 --request-timeout-ms 60000

# With RAG endpoints mounted (requires --features rag):
oxibonsai serve --model models/Ternary-Bonsai-1.7B.gguf --rag
```

> `oxibonsai serve` mounts the same bearer-auth + admission-control hardening
> shape as the standalone `oxibonsai-serve` binary (`src/cli/admission.rs`).
> For a richer, TOML-file-layered configuration surface (`[limits]`,
> `[auth]`, `[observability]`, etc. — useful when you
> want config-as-a-file rather than flags) see
> [`oxibonsai-serve`](#oxibonsai-serve--standalone-server). See also the
> [deployment guide](DEPLOYMENT.md) for reverse-proxy/TLS and production
> hardening guidance shared by both binaries.

---

### `info`

Display model metadata and architecture parsed from a GGUF file.

| Flag | Short | Type | Default | Help |
|------|-------|------|---------|------|
| `--model <PATH>` | `-m` | string | env `OXI_MODEL` | Path to the GGUF model file. Required unless `OXI_MODEL` is set. |
| `--json` | | bool | `false` | Emit info as JSON instead of human-readable text. |

```bash
oxibonsai info --model models/Ternary-Bonsai-1.7B.gguf
oxibonsai info --model models/Ternary-Bonsai-1.7B.gguf --json
```

---

### `benchmark`

Run a quick throughput benchmark on a tiny synthetic model. No real model
weights are required — useful for sanity-checking the runtime and SIMD/GPU
tiers.

| Flag | Short | Type | Default | Help |
|------|-------|------|---------|------|
| `--tokens <N>` | | usize | `100` | Total tokens to generate during the benchmark pass. |
| `--warmup <N>` | | usize | `10` | Warmup tokens generated before timing begins. |
| `--temperature <F>` | | f32 | `0.7` | Sampling temperature. |
| `--seed <N>` | | u64 | `42` | Random seed. |

```bash
oxibonsai benchmark --tokens 200 --warmup 20
```

---

### `quantize`

Quantize a GGUF model to a lower-precision format. This command performs the
**real** conversion end-to-end: it mmaps and parses the input GGUF, dequantizes
every source tensor to f32, re-encodes through the same
`oxibonsai_model::export::export_to_gguf` pipeline used by `oxibonsai convert`,
and writes the actual output bytes to `--output`. Reported tensor counts,
output size, and compression ratio are measured from the real write, not
estimated.

| Flag | Short | Type | Default | Help |
|------|-------|------|---------|------|
| `--input <PATH>` | | string | *(required)* | Path to the input GGUF model file. |
| `--output <PATH>` | | string | *(required)* | Destination path for the quantized file. |
| `--format <FMT>` | | string | `q1_0` | Target quantization format. Supported: `f32`, `q1_0`/`q1_0_g128`, `tq2_0_g128`/`ternary`, `fp8_e4m3`, `fp8_e5m2`, `q4_0`, `q8_0`, `q4_k`, `q5_k`, `q6_k`. An unrecognized value (e.g. `q2_k`, `q4_1`, `f16` output — not yet supported as export targets) fails fast with an error and no file is written. |

```bash
oxibonsai quantize --input models/model-f16.gguf --output models/model-q1_0.gguf --format q1_0
```

---

### `validate`

Validate that a GGUF file is well-formed and print a metadata summary.
Exits non-zero if the model config cannot be parsed.

| Flag | Short | Type | Default | Help |
|------|-------|------|---------|------|
| `--model <PATH>` | `-m` | string | *(required)* | Path to the GGUF model file to validate. |

```bash
oxibonsai validate --model models/Ternary-Bonsai-1.7B.gguf
```

---

### `convert`

Convert a HuggingFace safetensors model (or an ONNX `MatMulNBits` model) to
GGUF format.

| Flag | Short | Type | Default | Help |
|------|-------|------|---------|------|
| `--from <PATH>` | | string | *(required)* | Input directory containing `model.safetensors` (or shards) and `config.json`. With `--onnx`, this is the ONNX model file instead. |
| `--to <PATH>` | | string | *(required)* | Output GGUF file path. |
| `--quant <FMT>` | | string | `tq2_0_g128` | Quantization format: `tq2_0_g128` (default) or `q1_0_g128`. |
| `--onnx` | | bool | `false` | Treat `--from` as an ONNX model file (`MatMulNBits`, bits=2) and use the ONNX→GGUF converter. |

```bash
# HuggingFace unpacked safetensors → GGUF
oxibonsai convert \
  --from <unpacked-safetensors-dir> \
  --to models/Ternary-Bonsai-1.7B.gguf \
  --quant tq2_0_g128

# ONNX (MatMulNBits, bits=2) → GGUF
oxibonsai convert --onnx \
  --from path/to/model.onnx \
  --to models/Ternary-Bonsai-1.7B.gguf
```

---

### `eval`

Evaluate a model's generation quality (ROUGE-1/2/L) against a JSONL dataset
of `{"input": <prompt>, "expected_output": <reference>}` examples. Gated
behind the `eval` Cargo feature (`cargo build --features eval`; not part of
the default feature set).

The dataset is loaded and validated **before** the model is opened, so a
malformed `--dataset` fails fast without paying the GGUF load cost. Each
example is generated greedily (temperature 0) up to `--max-tokens`, then the
whole corpus is scored with `oxibonsai_eval::CorpusRouge`.

| Flag | Short | Type | Default | Help |
|------|-------|------|---------|------|
| `--model <PATH>` | `-m` | string | env `OXI_MODEL` | Path to the GGUF model file. Required unless `OXI_MODEL` is set. |
| `--dataset <PATH>` | | string | *(required)* | JSONL dataset path: one `{"input": "...", "expected_output": "..."}` object per line. |
| `--limit <N>` | | usize | all | Only evaluate the first N examples. |
| `--max-tokens <N>` | | usize | `128` | Maximum tokens generated per example. |
| `--max-seq-len <N>` | | usize | `4096` | Maximum sequence length (prompt + generated). |
| `--tokenizer <PATH>` | | string | env `OXI_TOKENIZER`, else auto-detect | Path to `tokenizer.json`. |
| `--report-json <PATH>` | | string | — | Also write the report as JSON to this path. |
| `--report-markdown <PATH>` | | string | — | Also write the report as Markdown to this path. |

```bash
oxibonsai eval --model models/Ternary-Bonsai-1.7B.gguf \
  --dataset eval_data.jsonl \
  --max-tokens 128 \
  --report-json report.json --report-markdown report.md
```

---

### `tokenizer`

Manage the Qwen3 tokenizer. Has two subcommands: `download` and `info`.

```text
oxibonsai tokenizer <download|info> [OPTIONS]
```

#### `tokenizer download`

Download `tokenizer.json` from HuggingFace and save it locally.

| Flag | Short | Type | Default | Help |
|------|-------|------|---------|------|
| `--output <PATH>` | | string | `models/tokenizer.json` | Destination path. Parent directories are created as needed. |
| `--repo <REPO>` | | string | `Qwen/Qwen3-8B` | HuggingFace repo to download from (must contain `tokenizer.json`). |
| `--force` | | bool | `false` | Overwrite an existing `tokenizer.json` without prompting. |

```bash
oxibonsai tokenizer download --output models/tokenizer.json
```

#### `tokenizer info`

Show the vocabulary size and model type stored in a `tokenizer.json`.

| Flag | Short | Type | Default | Help |
|------|-------|------|---------|------|
| `--path <PATH>` | | string | `models/tokenizer.json` | Path to `tokenizer.json`. |

```bash
oxibonsai tokenizer info --path models/tokenizer.json
```

---

## `oxibonsai-serve` — standalone server

A separate binary (crate `oxibonsai-serve`) that runs the OpenAI-compatible
HTTP server with a layered configuration system. Unlike the `oxibonsai serve`
subcommand, it parses argv by hand (no `clap`) and merges four configuration
layers: built-in defaults → TOML file (`--config`) → `OXIBONSAI_*` environment
variables → CLI flags (highest precedence).

```text
oxibonsai-serve [OPTIONS]
```

| Flag | Short | Type | Default | Help |
|------|-------|------|---------|------|
| `--config <PATH>` | | string | — | Path to a TOML configuration file (optional). |
| `--host <HOST>` | | string | `0.0.0.0` | Bind host. |
| `--port <PORT>` | | u16 | `8080` | Bind port (must be `1`–`65535`). |
| `--model <PATH>` | | string | — | Path to the GGUF model file. |
| `--tokenizer <PATH>` | | string | — | Path to a tokenizer file/directory (optional). |
| `--max-tokens <N>` | | usize | `256` | Default maximum tokens to generate per request. |
| `--temperature <F>` | | f32 | `0.7` | Default sampling temperature. |
| `--seed <N>` | | u64 | `42` | RNG seed for reproducible generation. |
| `--log-level <LEVEL>` | | string | `info` | Logging level: one of `error`, `warn`, `info`, `debug`, `trace`, `off`. |
| `--bearer-token <TOK>` | | string | — | Require the given bearer token on protected endpoints. |
| `--help` | `-h` | flag | — | Print help and exit. |
| `--version` | `-V` | flag | — | Print version and exit. |

> A flag is only treated as an explicit override when its value differs from
> the default, so passing a flag at its default value will not shadow a value
> set in the TOML file or environment.

Some configuration knobs are only reachable through TOML or the `OXIBONSAI_*`
environment variables (they have no CLI flag): `top_p`, `max_input_tokens`,
`max_concurrent_requests`, `engine_pool_size`, `per_request_timeout_ms`,
`metrics_enabled`, `metrics_path`, `tokenizer_kind`, and `quantization_hint`.
See [`OXIBONSAI_*`](#oxibonsai_-oxibonsai-serve-server).

```bash
oxibonsai-serve \
  --model models/Ternary-Bonsai-1.7B.gguf \
  --host 127.0.0.1 --port 8080 \
  --log-level info
```

TOML configuration (sections: `[bind]`, `[model]`, `[tokenizer]`,
`[sampling]`, `[limits]`, `[auth]`, `[observability]`, plus a top-level
`seed`):

```toml
[bind]
host = "127.0.0.1"
port = 8080

[model]
path = "models/Ternary-Bonsai-1.7B.gguf"
quantization_hint = "TQ2"

[tokenizer]
path = "models/tokenizer.json"
kind = "huggingface"

[sampling]
default_max_tokens = 256
default_temperature = 0.7
default_top_p = 1.0

[limits]
max_input_tokens = 8192
max_concurrent_requests = 32
engine_pool_size = 4
per_request_timeout_ms = 60000

[auth]
bearer_token = "secret"

[observability]
log_level = "info"
metrics_enabled = true
metrics_path = "/metrics"

seed = 42
```

---

## Environment variables

There are two distinct prefixes:

- **`OXI_*`** — read by the `oxibonsai` binary (and its `run`/`image`/`chat`/
  `serve`/`info` subcommands) to supply default file paths and toggles.
- **`OXIBONSAI_*`** — read by the standalone `oxibonsai-serve` server to
  populate its layered configuration.

### `OXI_*` (oxibonsai binary)

| Variable | Used by | Controls | Default |
|----------|---------|----------|---------|
| `OXI_MODEL` | `run`, `chat`, `serve`, `info` | GGUF model path (lets you omit `--model`). | — |
| `OXI_TOKENIZER` | `run`, `chat`, `serve` | `tokenizer.json` path/dir (lets you omit `--tokenizer`). | — (falls back to auto-detection) |
| `OXI_DIT_GGUF` | `image` | DiT (FLUX.2-Klein ternary) GGUF path. | `/tmp/parity.gguf` |
| `OXI_VAE_WEIGHTS` | `image` | VAE decoder weights — **either** a `.safetensors` file **or** a directory of per-tensor `.npy` exports. | `/tmp/bonsai_golden/vae/weights` |
| `OXI_TE_4BIT` | `image` | 2.1 GB 4-bit MLX text-encoder `model.safetensors`. Takes precedence over `OXI_TE_WEIGHTS`. | — |
| `OXI_TE_WEIGHTS` | `image` | Fallback text-encoder weights as an f32 `.npy` directory (used only when `--te` and `OXI_TE_4BIT` are both unset). | `/tmp/bonsai_golden/te/weights` |
| `OXI_TE_TOKENIZER_DIR` | `image` | Text-encoder tokenizer directory (contains `tokenizer.json`). | the TE dir (or the TE `.safetensors` file's parent) |
| `OXIBONSAI_ENGINE_POOL_SIZE` | `serve` | Number of engine replicas (also honoured by the standalone server, below). | `min(4, CPU cores)`; clamped to `1` on GPU/Metal |
| `OXIBONSAI_PROFILE_GPU` | `run` (macOS + `metal`) | When set (any value), prints a GPU profiling summary after generation. | unset (no profiling) |

> Resolution order for `image`'s text-encoder path: `--te` → `OXI_TE_4BIT` →
> `OXI_TE_WEIGHTS` → the built-in `.npy` default. A path ending in
> `.safetensors` selects the 4-bit MLX loader; otherwise it is read as an
> `.npy` directory.

### `OXIBONSAI_*` (oxibonsai-serve server)

All variables below are read by the standalone `oxibonsai-serve` binary and map
onto its layered configuration. Unrecognised `OXIBONSAI_*` variables are
ignored; recognised ones with malformed values are a hard error.

| Variable | Controls | Type | Default |
|----------|----------|------|---------|
| `OXIBONSAI_HOST` | Bind host. | string | `0.0.0.0` |
| `OXIBONSAI_PORT` | Bind port. | u16 | `8080` |
| `OXIBONSAI_MODEL_PATH` | GGUF model path. | path | — |
| `OXIBONSAI_TOKENIZER_PATH` | Tokenizer file/directory path. | path | — |
| `OXIBONSAI_TOKENIZER_KIND` | Tokenizer kind (e.g. `huggingface`, `oxitok`). | string | — |
| `OXIBONSAI_MAX_TOKENS` | Default max tokens to generate per request. | usize | `256` |
| `OXIBONSAI_TEMPERATURE` | Default sampling temperature. | f32 | `0.7` |
| `OXIBONSAI_TOP_P` | Default top-p (nucleus) sampling. | f32 | `1.0` |
| `OXIBONSAI_MAX_INPUT_TOKENS` | Maximum prompt length, in tokens. | usize | `8192` |
| `OXIBONSAI_MAX_CONCURRENT` | Maximum concurrent requests (HTTP admission). | usize | `32` |
| `OXIBONSAI_ENGINE_POOL_SIZE` | Number of inference-engine replicas. `None`/unset resolves to `min(4, CPU cores)` on CPU; clamped to `1` on GPU/Metal. Distinct from `MAX_CONCURRENT`. | usize | unset → `min(4, cores)` |
| `OXIBONSAI_REQUEST_TIMEOUT_MS` | Per-request timeout, in milliseconds. | u64 | `60000` |
| `OXIBONSAI_BEARER_TOKEN` | Bearer token required on protected endpoints. | string | — |
| `OXIBONSAI_LOG_LEVEL` | Log level (`error`/`warn`/`info`/`debug`/`trace`/`off`). | string | `info` |
| `OXIBONSAI_METRICS_ENABLED` | Enable Prometheus metrics. Accepts `true`/`yes`/`on`/`1` and `false`/`no`/`off`/`0` (case-insensitive). | bool | `true` |
| `OXIBONSAI_METRICS_PATH` | Path to serve Prometheus metrics at. | string | `/metrics` |
| `OXIBONSAI_SEED` | RNG seed for deterministic sampling. | u64 | `42` |
| `OXIBONSAI_QUANTIZATION_HINT` | Quantization hint (e.g. `TQ2`, `Q8_0`). | string | — |

### Precedence

For both binaries the effective value of any setting is resolved in this order
(highest wins):

```text
CLI flag  >  shell environment variable  >  .env file  >  built-in default
```

The `oxibonsai` binary loads `.env` (from the current directory or any parent)
without overriding already-set shell variables. The `oxibonsai-serve` binary
additionally layers a TOML `--config` file *below* the environment and CLI
layers:

```text
CLI flag  >  OXIBONSAI_* env  >  TOML --config file  >  built-in default
```
