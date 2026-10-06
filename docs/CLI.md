# OxiBonsai CLI Reference

This document is the command-line reference for the two OxiBonsai binaries:

- **`oxibonsai`** — the main binary: text generation, interactive chat, image
  generation, the OpenAI-compatible server, model tooling and downloads.
  Arguments are parsed with `clap`, so every subcommand supports `--help`.
- **`oxibonsai-serve`** — a standalone OpenAI-compatible HTTP server with its
  own argument parser and a layered TOML / environment / flag configuration.

The tables below follow `oxibonsai <command> --help` and `oxibonsai-serve
--help` of this release. The sources of truth are `src/cli/args.rs` (the `clap`
definitions for `oxibonsai`) and `crates/oxibonsai-serve/src/{args.rs,env.rs,
config.rs,hardening.rs}` (the standalone server); when a build disagrees with
this page, the build's own `--help` is authoritative. For which models run and
how they are told apart, see [`docs/models.md`](models.md); for running the
servers in production, see the [deployment guide](DEPLOYMENT.md).

## Contents

- [`oxibonsai` — global usage](#oxibonsai--global-usage)
  - [Build features](#build-features)
  - [Configuration file (`--config`)](#configuration-file---config)
  - [Precedence](#precedence)
  - [`run`](#run)
  - [`image`](#image)
  - [`repl`](#repl)
  - [`chat`](#chat)
  - [`serve`](#serve)
  - [`info`](#info)
  - [`build-info`](#build-info)
  - [`benchmark`](#benchmark)
  - [`quantize`](#quantize)
  - [`validate`](#validate)
  - [`convert`](#convert)
  - [`eval`](#eval)
  - [`tokenizer`](#tokenizer)
  - [`pull`](#pull)
- [`oxibonsai-serve` — standalone server](#oxibonsai-serve--standalone-server)
- [HTTP surface shared by both servers](#http-surface-shared-by-both-servers)
- [Environment variables](#environment-variables)

---

## `oxibonsai` — global usage

```text
oxibonsai [--config <PATH>] <COMMAND> [OPTIONS]
```

| Command | What it does |
|---------|--------------|
| [`run`](#run) | One-shot generation from a prompt, streamed to stdout. |
| [`chat`](#chat) | Interactive multi-turn conversation. |
| [`serve`](#serve) | OpenAI-compatible API server (needs the `server` feature, on by default). |
| [`image`](#image) / [`repl`](#repl) | Bonsai-Image text-to-image, one-shot and interactive. |
| [`info`](#info) / [`build-info`](#build-info) / [`validate`](#validate) | Inspect a GGUF; inspect the binary; check a GGUF is well-formed. |
| [`benchmark`](#benchmark) | Throughput benchmark of a real model (or an explicitly synthetic toy). |
| [`quantize`](#quantize) / [`convert`](#convert) | Re-encode a GGUF; convert safetensors or ONNX to GGUF. |
| [`eval`](#eval) | Evaluation harness against a JSONL dataset (needs the `eval` feature). |
| [`tokenizer`](#tokenizer) | Download or inspect `tokenizer.json`. |
| [`pull`](#pull) | Download a Bonsai 2 / Bonsai model artifact with fail-closed verification. |

| Flag | Help |
|------|------|
| `--config <PATH>` | Path to an OxiBonsai TOML configuration file; global, accepted by every subcommand. See [Configuration file](#configuration-file---config). |
| `-h`, `--help` | Print help (also on every subcommand). |
| `-V`, `--version` | Print the version. |

At start-up `oxibonsai` loads a `.env` file from the current directory or any
parent directory. It never overrides a variable that is already set in the real
environment, so a value exported in the shell wins over the same key in `.env`.

Every error exits non-zero with a message on stderr. Typed refusals carry a
stable code (for example `NOT_A_HYBRID_MODEL`, `BACKEND_UNAVAILABLE`,
`vision_vocabulary_mismatch`) so a script can match on it. Log lines go to
stderr; generated text goes to stdout. `RUST_LOG` overrides the log filter;
otherwise `[observability].log_level` from `--config` applies (default
`info`), and `[observability].json_logs = true` switches to JSON lines.

### Build features

`oxibonsai` is built from the repository root package (`cargo install
oxibonsai-cli`). Every GPU, vision-adjacent and optional subsystem is a Cargo
feature; the default build is CPU-only Pure Rust.

| Feature | Effect | Default |
|---------|--------|---------|
| `server` | The `serve` subcommand. | on |
| `rag` | `serve --rag` (the `/rag/*` endpoints). | off |
| `eval` | The `eval` subcommand. | off |
| `metal` | Apple Silicon GPU backend (`--backend metal`, the Metal hybrid runner, the Metal vision tower, GPU image stages). | off |
| `native-cuda` | NVIDIA CUDA backend (NVRTC kernels). This release's CUDA code was not exercised on CUDA hardware and is **unvalidated**; see the README's Known Limitations. | off |
| `hf-tokenizer` | The HuggingFace `tokenizers` backend, selectable with `--tokenizer-backend hf`. The Pure-Rust native tokenizer is always available and is what `auto` picks in a default build. | off |
| `simd-avx2`, `simd-avx512`, `simd-neon` | Empty compatibility features: every CPU tier (scalar, AVX2, AVX-512, NEON) is always compiled in and chosen at run time by CPU feature detection, so these change nothing. | off |

`oxibonsai build-info` prints which of these the binary in hand was built
with.

### Configuration file (`--config`)

`--config <PATH>` loads a flat TOML file with five sections. It is strict: an
unreadable or unparsable file is a hard error naming the path, and an unknown
section or key is an error too (a typo'd key is never silently ignored).

| Section | Keys |
|---------|------|
| `[sampling]` | `temperature`, `top_k`, `top_p`, `min_p`, `repetition_penalty`, `frequency_penalty`, `presence_penalty`, `max_tokens`, `seed` |
| `[model]` | `model_path`, `tokenizer_path`, `max_seq_len`, `backend`, `rope_scaling`, `reasoning_effort`, `enable_thinking`, `prefill_chunk`, `ptq1_transcode` |
| `[server]` | `host`, `port`, `cuda_device`, `bearer_token_file`, `rate_limit_rpm`, `rate_limit_burst`, `cors_origin`, `max_body_bytes`, `max_output_tokens`, `enable_ui` |
| `[observability]` | `log_level`, `json_logs` |
| `[imagen]` | `model_path`, `width`, `height`, `steps`, `seed`, `output_dir` (`guidance_scale` is refused: the FLUX.2 Klein DiT is guidance-distilled and this pipeline has no classifier-free-guidance stage) |

There is no `bearer_token` key: a secret belongs in a file named by
`[server].bearer_token_file`, in `OXIBONSAI_BEARER_TOKEN`, or in the environment
of the service manager.

```toml
[model]
model_path = "models/Ternary-Bonsai-2-27B-PQ2_0.gguf"
max_seq_len = 8192
backend = "auto"

[sampling]
temperature = 0.6
max_tokens = 512

[server]
host = "127.0.0.1"
port = 8080
bearer_token_file = "/etc/oxibonsai/token"
rate_limit_rpm = 120

[observability]
log_level = "info"
```

A value from the file is used only where the matching flag was not passed, and
a key that is absent from the file is absent: it falls through to the model's
own `general.sampling.*` default and then to the built-in one, never to a
hidden struct default. A `[server].host` from the file can never widen the
bind address when `--host` was passed explicitly.

### Precedence

For the `oxibonsai` binary, each setting is resolved through the links of this
chain that apply to it, highest first:

```text
CLI flag  >  --config file  >  environment (shell, then .env)  >  model-declared default  >  built-in default
```

- Sampling values (`temperature`, `top_k`, `top_p`, `min_p`) have no environment
  variable: flag, then `[sampling]`, then the GGUF's `general.sampling.*` keys
  (Bonsai 2: temperature 1.0, top-p 0.95, top-k 20), then the built-ins 0.7 /
  0.9 / 40 and `min_p` 0.0. The repetition, frequency and presence penalties
  default to the no-op values (1.0, 0.0, 0.0) everywhere; nothing hidden is
  ever applied.
- The model path is `--model`, then `[model].model_path`, then `OXI_MODEL`; the
  tokenizer is `--tokenizer`, then `[model].tokenizer_path`, then
  `OXI_TOKENIZER`, then auto-detection next to the model.
- The image assets are `--dit` / `--vae` / `--te` / `--tokenizer`, then (for the
  DiT) `[imagen].model_path`, then `OXI_DIT_GGUF` / `OXI_VAE_WEIGHTS` /
  `OXI_TE_4BIT` / `OXI_TE_WEIGHTS` / `OXI_TE_TOKENIZER_DIR`, and as a last resort
  a file that already exists in the per-user data directory (see
  [`image`](#image)). There is no hard-wired default path.
- The server hardening knobs are the flag, then `[server]`, then the matching
  `OXIBONSAI_*` variable, then the built-in default.

The standalone `oxibonsai-serve` binary orders its layers differently — the
environment sits **above** its TOML file; see
[its section](#oxibonsai-serve--standalone-server).

---

### `run`

Run inference on a GGUF model and stream the generated text to stdout.

```bash
oxibonsai run \
  --model models/Ternary-Bonsai-1.7B.gguf \
  --prompt "Explain ternary quantization in one sentence." \
  --max-tokens 256 --temperature 0.7
```

**Model and prompt**

| Flag | Default | Help |
|------|---------|------|
| `-m, --model <PATH>` | `[model].model_path`, else env `OXI_MODEL` | GGUF model file. |
| `-p, --prompt <TEXT>` | *(required)* | Prompt text; `-` reads it from stdin. |
| `--max-tokens <N>` | `256`, or `[sampling].max_tokens` | Maximum tokens to generate; must be at least 1. |
| `--max-seq-len <N>` (alias `--ctx`) | `8192` for a Bonsai 2 `qwen35` model, `4096` otherwise, or `[model].max_seq_len` | Context window (prompt + generated); at least 1. For a `qwen35` model it is refused above both the model's declared context and the RAM-derived bound. |
| `--tokenizer <PATH>` | `[model].tokenizer_path`, else env `OXI_TOKENIZER`, else auto-detected next to the model | Path to `tokenizer.json`. |
| `--tokenizer-backend <B>` | `auto` | `auto`, `native` (Pure-Rust BPE, always available on every target) or `hf` (needs the `hf-tokenizer` build feature). |
| `--allow-vocab-mismatch` | off | Proceed when the resolved tokenizer's vocabulary is *smaller* than the model's (TOK-08). A smaller vocabulary is a different BPE and silently mis-tokenizes every prompt; pass it only if you verified the tokenizer. A *larger* vocabulary is always a hard error. |
| `--no-stream` | off | Print the whole completion once at the end instead of streaming (streaming is the default on every backend). |

**Sampling**

| Flag | Default | Help |
|------|---------|------|
| `--temperature <F>` | flag, `[sampling]`, model `general.sampling.temp` (Bonsai 2: 1.0), else `0.7` | `0.0` is greedy argmax on every backend; must be at least 0. |
| `--top-k <N>` | flag, `[sampling]`, model (Bonsai 2: 20), else `40` | `0` disables. |
| `--top-p <F>` | flag, `[sampling]`, model (Bonsai 2: 0.95), else `0.9` | Nucleus sampling; in (0, 1]. |
| `--min-p <F>` | flag, `[sampling]`, model, else `0.0` | Applied after top-k and before top-p; only affects sampled decoding. |
| `--repetition-penalty <F>` | `1.0` | `1.0` disables; must be greater than 0. |
| `--frequency-penalty <F>`, `--presence-penalty <F>` | `0.0` | OpenAI-style, range [-2.0, 2.0], applied over the generated history on every backend. |
| `--seed <N>` | flag, `[sampling].seed`, else `42` | The same seed, model, prompt and sampling flags produce byte-identical output. |

With `--grammar` or `--stop`, generation runs through a dedicated
token-by-token loop that cannot apply a repetition, frequency or presence
penalty; combining one of those flags with a non-default penalty is a hard
error rather than a silently dropped flag.

**Constrained generation**

| Flag | Help |
|------|------|
| `--grammar <FILE>` | Constrain output to a grammar: a `.gbnf` file, or any other extension parsed as a JSON Schema. Works on every backend. |
| `--stop <STRING>` | Stop as soon as the string appears in the decoded output; repeatable. |

**Chat contract** (these only mean something inside the model's chat template)

| Flag | Help |
|------|------|
| `--chat` | Render `--prompt` as one user turn through the GGUF's own `tokenizer.chat_template` (the built-in ChatML/Qwen3 template when the file ships none). The four rows below require it, and `--image` implies it. |
| `--think`, `--no-think` | Set `enable_thinking` in the template. Neither flag = the template's own default (the Bonsai 2 template thinks by default). |
| `--reasoning-effort <E>` | `low`, `medium` or `xhigh` (the Bonsai 2 contract). |
| `--tools <FILE>` | JSON file with an OpenAI-style `tools` array, passed to the template as raw JSON text so key order and number formatting survive byte for byte. |
| `--show-reasoning`, `--hide-reasoning` | Print the `<think>` block to stderr while the answer streams to stdout (the default), or drop it. |

**Backend and numerics**

| Flag | Default | Help |
|------|---------|------|
| `--backend <B>` | `auto` | `auto` takes the best available — a GPU when this build has one and the host serves the model (for a hybrid `qwen35` model such as Bonsai 2, the Metal hybrid runner), else the best CPU SIMD tier. `cpu` forces the CPU tier, including for `--temperature 0`. `metal` demands the Metal GPU and fails with a typed non-zero error — never a silent CPU fallback — when this build or host cannot serve the model. A binary built without the `metal` feature has no GPU backend. |
| `--rope-scaling <MODE>` | `auto` | `auto` honours the `<arch>.rope.scaling.*` keys the GGUF declares, as llama.cpp does (Bonsai-8B declares YaRN factor 4); `off` forces plain RoPE and reproduces OxiBonsai 0.2.3 and earlier; `on` requires the file to declare scaling and errors otherwise. |
| `--ptq1-transcode` | off | Transcode every PTQ1_0 (1.75-bit) matrix to the lossless 2-bit PQ2_0 layout at load, in anonymous RAM (about 7.2 GB for the 27B) instead of mmapping the native file. A no-op, with a log line, for a file with no PTQ1_0 tensor. |
| `--prefill-chunk <N>` | `512` for a `qwen35` hybrid; the model's own plan for a dense one | Prompt-ingestion chunk in tokens; at least 1. On the Metal hybrid runner the log reports the size actually in effect, which is smaller than asked when the KV window's memory budget cannot hold it. |

**Vision** (a Bonsai 2 `qwen35` model reading images)

| Flag | Default | Help |
|------|---------|------|
| `--mmproj <PATH>` | — | Vision projector GGUF (`clip` architecture, for example `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf`). Loaded once, for the executor the engine decodes on: the Metal tower (the file's weights as stored, about 0.87 GiB) beside the Metal hybrid runner, or the CPU tower (about 1.7 GiB of `f32` weights) beside the CPU model. Refused for any other architecture (`NOT_A_HYBRID_MODEL`, before the model is loaded) and for a projector the tower does not recognise. |
| `--image <REF>` | — | Repeatable; needs `--mmproj`; implies `--chat`. A PNG or JPEG file, a base64 `data:image/...;base64,` URI, or — only with `--allow-image-url-fetch` — an `http(s)` URL. Each image becomes a `<\|vision_start\|><\|image_pad\|><\|vision_end\|>` part of the user turn, ahead of the prompt text. |
| `--image-max-tokens <N>` | `1024` | Per-image merged-token budget, `1..=16384`. An image is smart-resized to multiples of 32 pixels (Pillow bicubic, aspect preserved) and downscaled until its `(H/32) * (W/32)` grid fits; one that cannot fit even at its smallest aspect-preserving size is refused (`image_too_many_tokens`). It never upscales. |
| `--allow-image-url-fetch` | off | Fetch remote `http(s)` image references (see [Remote images](#remote-images)); also `OXI_ALLOW_IMAGE_URL_FETCH=1` (the flag wins). Without it a remote reference is refused with `image_url_fetch_disabled` and nothing is opened or resolved. With it each image is one `GET` (no body, cookies, credentials or retries; `User-Agent: oxibonsai/<version>`), at most 3 redirects, a `200` answer of at most 32 MiB and one deadline per image, from **public addresses only**: a host that is, or resolves to, a loopback, private, link-local, multicast or other special-purpose address, or that is named `localhost`, is refused (`image_url_refused`) unless `--image-url-allow-host` names it. |
| `--image-url-timeout-ms <MS>` | `10000` | Per-image deadline of a remote fetch, covering name resolution, connect, TLS, the response head and the whole body; at least 1. Also `OXI_IMAGE_URL_TIMEOUT_MS` (the flag wins; `0` or a non-number is a start-up error naming the setting). Needs `--allow-image-url-fetch`. |
| `--image-url-allow-host <HOST[:PORT]>` | — | Repeatable. Exempts a host from the public-address rule — an exact, case-insensitive match of the URL's host (an IP literal by its address, whatever spelling), and of its port when given — for an intranet image store or a loopback test server. Nothing else is relaxed. Also `OXI_IMAGE_URL_ALLOW_HOSTS` (comma-separated); the flag's entries are **added** to the variable's. A malformed entry is a start-up error naming it; an allowlist without `--allow-image-url-fetch` is a start-up error. |

```bash
oxibonsai run --model models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  --mmproj models/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf \
  --image photo.png --prompt "Describe this image briefly." --temperature 0
```

The default `--image-max-tokens` is 1024, the Bonsai demo's own; the PrismML
llama.cpp fork's `llama-server` accepts up to 4096 image tokens per image by
default (about 2048 x 2048 against about 1024 x 1024). The same large photo is
therefore downscaled further here, which means fewer image rows to prefill but
less detail for tasks that read fine print. Pass `--image-max-tokens 4096` for
the fork's resolution; the default is deliberately not changed.

`--image-max-tokens` also bounds what a *source* image may cost to decode: an
image with more pixels than 16 times what its token budget can use (never below
16 megapixels, never above 8192 x 8192) is refused as `image_too_large` from its
header, before it is inflated. At the default budget that is 16 777 216 pixels:
a 4032 x 3024 phone photo passes; a 48-megapixel one needs `--image-max-tokens
3072` or a smaller picture.

Image rows are prefilled on the executor the engine decodes on. With `--mmproj`
and `--backend auto` on a Metal host the Metal hybrid runner and the Metal tower
are kept: both hybrid executors (Metal and CPU) can prefill image rows, so
nothing is rebuilt on the CPU. Measured on an M3 with 24 GB (a loaded host, so
read these as indicative), the reference 256 x 192 image prompt takes about 14 s
end to end through `oxibonsai run` on the Metal hybrid runner with the Metal tower
and about 78 s with `--backend cpu`; the vision-tower encode alone is 0.12 to
0.13 s for 256 x 192 and 2.6 to 2.7 s for 768 x 768 on Metal, against 1.8 to 6 s
and 23 to 30 s on the CPU tower. On the CPU model the prefill of the image rows
dominates: 54.9 s for a 67-row prompt and 32.6 s for a 75-row one (measured on
separate runs at different host load; order of magnitude only — the larger prompt
was the faster run), plus about 5 s of vision encode. A prompt whose expanded length (image rows included) does not
fit `--max-seq-len`, or fills it exactly so that no token could be generated, is
refused before the vision encode.

Loading the projector also checks the language model's own vocabulary:
`<|vision_start|>`, `<|vision_end|>` and `<|image_pad|>` must sit at the ids the
image splice looks for (248053, 248054 and 248056 in the Bonsai 2 vocabulary).
A model whose vocabulary lacks one, places one elsewhere, or carries no
vocabulary at all is refused at start-up — before any language-model weight is
bound — with `[vision_vocabulary_mismatch]`, naming every wrong token with the
id the vocabulary gives it and the id the splice uses. A text-only command never
consults it.

---

### `image`

Generate an image from a text prompt with Bonsai-Image (FLUX.2 Klein): text
encoder, DiT, VAE, PNG, entirely in Pure Rust. The end-to-end walkthrough,
asset downloads and timings are in [`docs/IMAGEN.md`](IMAGEN.md).

| Flag | Default | Help |
|------|---------|------|
| `-p, --prompt <TEXT>` | *(required)* | Text prompt; `-` reads stdin. |
| `-o, --out <PATH>` | *(required)* | Output PNG path. A relative path is placed under `[imagen].output_dir` when `--config` sets one. |
| `--seed <N>` | `42`, or `[imagen].seed` | Seed of the initial noise (an MLX-exact Threefry port: `--seed 42` reproduces the official mflux sample). |
| `--steps <N>` | `4`, or `[imagen].steps` | Euler sampler steps. |
| `--width <N>`, `--height <N>` | `512`, or `[imagen].width` / `[imagen].height` | Image size in pixels. |
| `--dit <PATH>` | `[imagen].model_path`, else env `OXI_DIT_GGUF` | DiT GGUF. **Required** from one of those sources or the data-directory fallback below. |
| `--vae <PATH>` | env `OXI_VAE_WEIGHTS` | VAE weights: a `.safetensors` file or a directory of per-tensor `.npy` exports; the loader picks the reader from the path. **Required**, as above. |
| `--te <PATH>` | env `OXI_TE_4BIT`, else `OXI_TE_WEIGHTS` | Text encoder: a 4-bit MLX `model.safetensors` file (a path ending in `.safetensors` selects that loader) or an f32 `.npy` directory. **Required**, as above. |
| `--tokenizer <PATH>` | env `OXI_TE_TOKENIZER_DIR`, else the TE directory | Directory containing `tokenizer.json`; for a `.safetensors` `--te` the file's parent directory. |

There is no hard-wired default path. When neither the flag nor the variable is
set, the command uses a file that **already exists** at `models/dit.gguf`,
`models/vae` or `models/te` under the per-user data directory
(`$XDG_DATA_HOME/oxibonsai`, else `~/Library/Application Support/oxibonsai` on
macOS, `~/.local/share/oxibonsai` on other Unix, `%APPDATA%\oxibonsai` on
Windows); otherwise it stops with an error naming the flag, the variable and
that location. Nothing is ever read from `/tmp` or another world-writable
directory by default.

```bash
oxibonsai image --prompt "a tiny bonsai tree in a ceramic pot" --out bonsai.png \
  --dit bonsai-dit.gguf --vae bonsai-vae/diffusion_pytorch_model.safetensors \
  --te bonsai-te/model.safetensors --seed 42 --steps 4
```

With `OXI_DIT_GGUF`, `OXI_VAE_WEIGHTS` and `OXI_TE_4BIT` set (in the shell or in
`.env`) the path flags can be omitted. On a Metal build the GPU stages are on by
default and each can be opted out of; see
[Environment variables](#environment-variables).

---

### `repl`

Interactive image generation: loads the DiT, VAE and text encoder **once** and
renders the prompts you type without re-loading weights. On a kitty-graphics
terminal (Ghostty) each image is shown inline; elsewhere it is written to a file
and optionally opened in a viewer. Model paths resolve exactly like
[`image`](#image): flag, then environment, then the data-directory fallback,
otherwise an error.

| Flag | Default | Help |
|------|---------|------|
| `--seed <N>`, `--steps <N>`, `--width <N>`, `--height <N>` | `42`, `4`, `512`, `512` (or `[imagen]`) | Initial parameters, changeable inside the session. |
| `--cpu-te` | off | Run the text-encoder GEMM on the CPU instead of the Metal GPU (without it, `repl` sets `OXI_TE_GPU=1` unless you already set it). |
| `--dit <PATH>`, `--vae <PATH>`, `--te <PATH>`, `--tokenizer <PATH>` | as for `image` | Asset paths, resolved exactly as for `image` (flag, environment, data-directory fallback, else an error). |

Text-encoder residency is source-aware: a 4-bit MLX `model.safetensors` (about
2.1 GB on disk) is **not** kept dequantised between prompts by default, while an
f32 `.npy` directory (about 15 GB) already is. `OXI_TE_RESIDENT=1` pins the
fully dequantised f32 text encoder (about 16 GB) in memory for the whole
session; `OXI_TE_RESIDENT=0` forces it transient. Pin it only on a high-memory
machine.

Inside the session a bare line is a prompt and a `:`-prefixed line is a command:

| Command | Effect |
|---------|--------|
| `:fast` | Preset: 2 steps, 384x384. |
| `:hq` | Preset: 8 steps, 512x512. |
| `:steps N`, `:seed N` | Set one parameter. |
| `:size WxH` | Output size (`:size N` for a square). |
| `:out PATH` | Write to `PATH` (no argument = an automatic `oxibonsai-repl-NNN.png`). |
| `:open on` / `:open off` | Open each image in a viewer (non-inline terminals). |
| `:show` (alias `:set`) | Print the current settings. |
| `:help`, `:quit` | Command reference; exit (also `:q`, `:exit` and Ctrl-D). |

---

### `chat`

Interactive multi-turn conversation. `quit` or `exit` (or Ctrl-D) leaves; `/reset`
clears the conversation (and re-attaches any `--image` to the next message);
Ctrl-C interrupts the current generation without exiting. The whole
conversation is re-rendered through the chat template every turn, so the
oldest turns are dropped once it no longer fits `--max-seq-len`.

`chat` takes the same flags as [`run`](#run) with these differences: there is no
`--prompt` and no `--no-stream`; `--max-tokens` defaults to `512` per turn;
`--think`, `--no-think`, `--reasoning-effort`, `--tools`, `--show-reasoning` and
`--hide-reasoning` apply to every turn (the contract is always on); `--image`
attaches to the **first** user message of the session and is re-sent with every
turn while that message stays in the context window.

```bash
oxibonsai chat --model models/Bonsai-8B.gguf --temperature 0.7
```

---

### `serve`

Start an OpenAI-compatible API server backed by the in-process engine. It needs
the `server` feature (on by default). The server mounts the same hardening
stack as the standalone binary — admission control, body limit, rate limit,
bearer auth, CORS, in that order, innermost first — and exposes every knob as a
flag that wins over its `OXIBONSAI_*` environment variable.

```bash
oxibonsai serve --model models/Ternary-Bonsai-1.7B.gguf          # 127.0.0.1:8080

# Reachable beyond loopback: a token is mandatory (see "Bind safety" below).
oxibonsai serve --model models/Ternary-Bonsai-1.7B.gguf --host 0.0.0.0 \
  --bearer-token-file /etc/oxibonsai/token --rate-limit-rpm 120

# With the RAG endpoints (needs --features rag):
oxibonsai serve --model models/Ternary-Bonsai-1.7B.gguf --rag
```

| Flag | Default | Help |
|------|---------|------|
| `-m, --model <PATH>` | `[model].model_path`, else env `OXI_MODEL` | GGUF model. |
| `--host <HOST>` | `127.0.0.1`, or `[server].host` | Bind address. |
| `--port <PORT>` | `8080`, or `[server].port` | Port. |
| `--max-seq-len <N>` (alias `--ctx`) | `8192` for `qwen35`, else `4096` | Per-request context window (prompt + generated); refused for a `qwen35` model above the model's declared context or the RAM-derived bound. `GET /v1/models` reports the window actually in force as `max_context_length`. |
| `--tokenizer <PATH>` | `[model].tokenizer_path`, else env `OXI_TOKENIZER`, else auto-detected | Tokenizer; same vocabulary-compatibility check as `run`. |
| `--pool-size <N>` | env `OXIBONSAI_ENGINE_POOL_SIZE`, else the sizing rule below | Number of engine replicas. |
| `--bearer-token <TOKEN>` | env `OXIBONSAI_BEARER_TOKEN` | Bearer token required on every endpoint except `/health`, `/metrics` and `/ui/health` (constant-time comparison; at least 16 bytes). **Visible to other local users through `ps`**, and a start-up warning says so; prefer the file or the environment. This flag does **not** gate `/admin/*` (see below). |
| `--bearer-token-file <PATH>` | env `OXIBONSAI_BEARER_TOKEN_FILE`, or `[server].bearer_token_file` | File whose trimmed contents are the bearer token; mutually exclusive with `--bearer-token`, and it overrides a token from the environment. |
| `--max-concurrent-requests <N>` | `32` | Requests admitted at once; beyond it, `503` with `Retry-After`. The effective ceiling is `min(N, 4 x pool size)`, never below 1. `/health`, `/readyz`, `/metrics` and `/ui/health` are answered outside this budget and keep answering at the limit; `GET /v1/models` and every other route are inside it. |
| `--request-timeout-ms <N>` | `60000` | Per-request deadline. See [Timeouts](#timeouts-and-image-requests). |
| `--max-body-bytes <N>` | env `OXIBONSAI_MAX_BODY_BYTES`, else 4 MiB | Request-body ceiling; above it, `413`. |
| `--max-output-tokens <N>` | `8192`, or `[server].max_output_tokens` | Hard ceiling on a request's effective `max_tokens` / `max_completion_tokens`; above it, `400` naming the ceiling. |
| `--rate-limit-rpm <N>` | env `OXIBONSAI_RATE_LIMIT_RPM`, unset = disabled | Requests per minute per client IP before a `429`. |
| `--rate-limit-burst <N>` | env `OXIBONSAI_RATE_LIMIT_BURST`, else `20` | Burst allowance on top of the rate. |
| `--cors-origin <ORIGIN>` | env `OXIBONSAI_CORS_ORIGIN`, unset = **no CORS headers** | One allowed origin (the environment variable takes a comma-separated list). |
| `--cors-allow-credentials` | env `OXIBONSAI_CORS_ALLOW_CREDENTIALS` | Send `Access-Control-Allow-Credentials: true`; needs a specific origin, never `*`. |
| `--enable-ui` | off | Mount the bundled chat UI at `GET /ui`. It sits behind the same bearer auth as everything else (only `/ui/health` is exempt); with no token configured it is open. |
| `--backend`, `--rope-scaling`, `--ptq1-transcode`, `--prefill-chunk` | as for `run` | Applied to every replica. |
| `--think`, `--no-think`, `--reasoning-effort <E>`, `--tools <FILE>` | — | Server-wide defaults a chat request inherits when it carries none of its own (`enable_thinking` as `chat_template_kwargs.enable_thinking` or top level, `reasoning_effort`, `tools`); a request always overrides them. |
| `--rag` | off | Also mount `/rag/index`, `/rag/query`, `/rag/stats`. Needs the `rag` feature. `/rag/query` runs under `--request-timeout-ms` like the generation routes and is cancelled when its client disconnects (see [Timeouts](#timeouts-and-image-requests)). |
| `--embedding-backend <B>` | `model` | Which backend answers `/v1/embeddings`: `model` (mean-pooled hidden states of the loaded model, dense and hybrid alike; a hybrid model embeds on the CPU model whatever `--backend` says), `none` (always `501`) or `tfidf` (lexical vectors fitted once at start-up on `--embedding-corpus`). |
| `--embedding-corpus <FILE>` | — | UTF-8 file, one document per line; required by and only valid with `tfidf`. |
| `--cuda-device <N>` | `[server].cuda_device` | CUDA device ordinal (sets `OXIBONSAI_CUDA_DEVICE` before the async runtime starts); only meaningful on a native-CUDA build with several devices. |
| `--mmproj <PATH>`, `--image-max-tokens <N>`, `--allow-image-url-fetch`, `--image-url-timeout-ms <MS>`, `--image-url-allow-host <HOST[:PORT]>`, `--media-path <DIR>` | — | Vision; see below. `--image` is refused: a server receives images in each request. |

**The model's checksum is verified before it loads.** When the checksum
manifest lists the model file, its SHA-256 must match or the server refuses to
start. The manifest defaults to `scripts/checksums.sha256` resolved relative to
the **current working directory** (so it is found only when started from the
repository root); set `OXIBONSAI_CHECKSUMS_FILE=<path>` to point elsewhere. No
manifest, or one that does not list the file, means no check.

**Bind safety.** A non-loopback `--host` with no bearer token configured is
refused at start-up. To acknowledge the risk explicitly, set
`OXIBONSAI_INSECURE_NO_AUTH=1` (the subcommand has no flag for it; the
standalone binary has `--insecure-no-auth`). A token read from
`--bearer-token-file` counts as a configured token.

**`/admin/*` has its own credential.** `--bearer-token` does not gate it. The
admin surface is always authenticated by its own token — `OXIBONSAI_ADMIN_TOKEN`
or `OXI_ADMIN_TOKEN` — and every `/admin/*` request is answered `403` while
neither is set, independent of the bearer token.

**Seed.** There is no `--seed` flag: the server's sampler seed is
`OXIBONSAI_SEED` when it parses as an integer, else `[sampling].seed` from
`--config`, else a fresh pseudo-random value at each start. A request's own
`seed` overrides it for that request.

**Pool size.** `--pool-size`, else `OXIBONSAI_ENGINE_POOL_SIZE`, else:

| Model and tier | Default | An explicit size is |
|----------------|---------|---------------------|
| Dense, CPU tier (also `--backend cpu`) | `min(4, CPU cores)` | honoured |
| Dense, Metal | `1` | capped at `OXIBONSAI_METAL_MAX_SESSIONS` (default `4`; a memory bound — each replica's device KV cache is about 604 MB for the 8B at a 4096 context) |
| Hybrid `qwen35`, either executor | `1` | honoured on the CPU, capped as above on Metal |
| CUDA-only build | `1` | clamped to `1` (the CUDA graph is still a process-global singleton) |

Replicas share one token-embedding table and the weight mapping, so an extra
replica costs its own KV cache (for a hybrid model also its recurrent state and
scratch), not a copy of the model. They buy isolation and latency fairness,
**not** GPU throughput: measured on an M3 with the real Ternary-Bonsai-1.7B and
eight concurrent greedy requests, pools of 1, 2 and 3 served the batch at 52.2,
54.2 and 54.5 tok/s aggregate.

#### Serving images

With `--mmproj`, `image_url` content parts on `/v1/chat/completions` and
`/v1/chat/completions/extended` are resolved as follows: base64 `data:` URIs
always; `file://` references only inside the media directory (`--media-path`,
else `OXI_MEDIA_PATH`) — relative paths, no `..`, and the resolved file must
stay inside the directory once symlinks are followed, otherwise
`400 image_file_refused`; remote `http(s)` URLs only with
`--allow-image-url-fetch`, under the address policy of
[Remote images](#remote-images) (without it, `400 image_url_fetch_disabled`
before anything is opened). The media directory must exist, and with
`--mmproj` it is checked before the model loads. Both chat endpoints stream an
image request exactly as they answer it without `stream`. Without `--mmproj` an
image request is `400 vision_unavailable` (a dense model answers `400
NOT_A_HYBRID_MODEL` either way).

**Image size and the request body.** A base64 `data:` URI travels inside the
JSON body, so it counts toward the body limit — `--max-body-bytes <N>` (default:
env `OXIBONSAI_MAX_BODY_BYTES`, else 4 MiB; a `413` above it) — and base64
inflates the image by 4/3: at the default a data URI carries about 3 MiB of
image, although the decoder accepts up to 32 MiB encoded. Raise
`--max-body-bytes` (about 43 MiB holds a 32 MiB image) or, with
`--allow-image-url-fetch`, let the server fetch a large image by URL instead.

#### Remote images

`--allow-image-url-fetch` (or `OXI_ALLOW_IMAGE_URL_FETCH=1`) makes `serve` fetch
an `http(s)` `image_url` — and `run` / `chat` an `--image https://...` — and
decode the bytes exactly like a data URI's (the same `prompt_tokens`). The
images of one request are fetched one after another; the fetches count toward
`--request-timeout-ms`, and a deadline that expires during one answers `504`
with `error.phase: image_fetch`, on both chat endpoints. A request that is over
— its client disconnected, or its deadline expired — drops the fetch in flight
within about 20 ms (the connection is closed) and starts no further one. The
process runs at most 8 fetches at once and lets 8 more wait for a slot (a wait
counts toward the per-image deadline); a fetch beyond that is refused at once,
before any connection, and the request answered `503 image_fetch_overloaded`
with `Retry-After` (see the codes below). The bound is compiled in, and `run` /
`chat` go through it too.

| Rule | What happens |
|---|---|
| URL | `http` or `https` (any case), a host, no `user:password@` (refused, and masked in the message), at most 2048 bytes, no whitespace or backslash; the fragment is never sent |
| Addresses | only public unicast addresses: every address a name resolves to must be public, or the whole fetch is refused (`image_url_refused`). Refused: loopback, private (`10/8`, `172.16/12`, `192.168/16`), `100.64/10`, link-local `169.254/16` (cloud metadata) and `fe80::/10`, `0.0.0.0/8`, documentation, benchmarking, multicast, reserved and broadcast ranges, `::`, every IPv4-mapped / IPv4-compatible IPv6 address, NAT64, Teredo, 6to4, unique-local `fc00::/7`, site-local, and IPv6 outside `2000::/3`. An IP literal is classified in every spelling (`2130706433`, `0x7f.1`, `[::ffff:127.0.0.1]`, ...); a zone id is refused; `localhost` and `*.localhost` are refused by name |
| DNS rebinding | names are resolved inside the HTTP client's resolver hook, which hands the client only the addresses it vetted: the addresses checked are the addresses dialled |
| Redirects | followed by the fetcher, at most 3; every hop is vetted again; `https` → `http` is refused; a relative `Location` resolves against the hop it came from; a redirect's body is never read |
| Request | one `GET`, no body, no cookies, no credentials, no retries, `User-Agent: oxibonsai/<version>`; no proxy (`HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY` are ignored) |
| Response | `200` only (any other status is `image_url_fetch_failed` naming it, its body never read); at most 32 MiB (`image_too_large`: a larger `Content-Length` before the body, a missing or lying one cut off past the cap); `Content-Type` is ignored — the decoder sniffs the bytes and applies the decode budget |
| TLS | an `https` certificate is verified against the public WebPKI roots built into the binary (the Mozilla set, as for `pull`; not the operating system's store): a store whose certificate chains only to a private CA fails as `connect/TLS` |
| Deadline | one per image over a wait for a transfer slot, resolution, connect, TLS, head and body: `--image-url-timeout-ms` (default 10000); a request that is over (client gone, deadline expired) drops its fetch at once |
| Bound | at most 8 transfers at once and 8 more waiting, process-wide (compiled in); beyond that a fetch is refused before any connection: `503 image_fetch_overloaded`, `Retry-After: 1` |
| Allowlist | `--image-url-allow-host <host[:port]>` / `OXI_IMAGE_URL_ALLOW_HOSTS` exempt exactly the named hosts (and ports) from the address rule and the `localhost` rule — nothing else; each redirect hop must itself be allowlisted or public |

```bash
# An intranet image store at 10.1.2.3:8080 (private, so it needs the allowlist);
# public image hosts need no entry.
oxibonsai serve --model models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  --mmproj models/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf \
  --allow-image-url-fetch --image-url-allow-host 10.1.2.3:8080 \
  --image-url-timeout-ms 5000 --request-timeout-ms 300000

curl -s http://127.0.0.1:8080/v1/chat/completions -H 'content-type: application/json' -d '{
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "http://10.1.2.3:8080/photos/cat.png"}},
    {"type": "text", "text": "Describe this image briefly."}]}],
  "max_tokens": 64}'
```

The start-up log says whether remote images are fetched, the deadline and the
allowlist; `oxibonsai info --mmproj` reports the same for the environment's
settings. Logs record the scheme, host and port, the status, the byte count and
the milliseconds of each fetch — never the path or query, never a resolved
address.

The settings are read only by a command given `--mmproj`, and checked there
before the model loads, each error naming its setting. Without the opt-in, an
allowlist (`--image-url-allow-host` or a non-empty `OXI_IMAGE_URL_ALLOW_HOSTS`)
or an `--image-url-timeout-ms` is a start-up error, because it would configure
nothing; `OXI_IMAGE_URL_TIMEOUT_MS` is read only once opted in. With the
opt-in, a malformed allowlist entry and a deadline (flag or variable) of `0` or
not a number are start-up errors. A text-only `run`, `chat` or `serve` never
consults `OXI_ALLOW_IMAGE_URL_FETCH`, `OXI_IMAGE_URL_TIMEOUT_MS` or
`OXI_IMAGE_URL_ALLOW_HOSTS`, so a stale value in the shell or in `.env` cannot
stop it (the flags themselves need `--mmproj`).

Every request is bounded by the engine's KV window, `min(the model's declared
context, --max-seq-len)`, reported by `GET /v1/models` as `max_context_length`;
text and image requests alike, on `/v1/chat/completions`,
`/v1/chat/completions/extended` and `/v1/completions`. An image counts as the
rows it expands to. A prompt that is not shorter than the window, or whose
length plus `max_tokens` does not fit it, is a typed `400
context_length_exceeded` naming the prompt's length, the window, the room left
and the larger context the model declares — before any vision encode is spent,
and never a `500` from inside the engine.

#### Image error codes

Every refusal of an image request is a `400` whose `error.code` names the
reason, with two exceptions: `image_encode_failed`, a `500` (the vision tower
failed on an image that was accepted), and `image_fetch_overloaded`, a `503`
with `Retry-After` and `error.type: overloaded_error` (the remote-image fetcher
was at capacity — a load shed, not a fault in the request; retry after the
delay). There are two spellings, on purpose:

| Spelling | Codes | Where they come from |
|---|---|---|
| `lower_snake_case` | `vision_unavailable`, `too_many_images`, `image_data_uri_invalid`, `image_url_scheme_unsupported`, `image_url_fetch_disabled`, `image_url_refused`, `image_url_fetch_failed`, `image_fetch_overloaded`, `image_file_refused`, `image_file_unreadable`, `image_format_unsupported`, `image_decode_failed`, `image_too_large`, `image_empty`, `image_too_many_tokens`, `image_placeholder_count_mismatch`, `image_placeholder_misplaced`, `image_grid_empty`, `image_prompt_overflow`, `image_config_invalid`, `image_encode_failed` | the **request**: what the client sent, or how the server was started (no projector) |
| `UPPER_SNAKE_CASE` | `NOT_A_HYBRID_MODEL`, `BACKEND_UNAVAILABLE` (reserved: no shipped executor raises it) | the **engine**: the stable codes of its own typed refusals, the same strings it reports everywhere else |

The four remote-image codes:

| Code | Status | Meaning |
|---|---|---|
| `image_url_fetch_disabled` | `400` | an `http(s)` reference while remote images are not enabled (no `--allow-image-url-fetch`), or — for a library caller — enabled with no fetcher installed; nothing was opened or resolved |
| `image_url_refused` | `400` | the address policy refused the URL or one of its redirect hops: scheme, credentials, length, a host that is not a public address (or `localhost`) and not allowlisted, an `https` → `http` redirect. The message names the host, never an address it resolved to |
| `image_url_fetch_failed` | `400` | a permitted fetch failed; the message names the step: `resolve`, `connect` (or `connect/TLS` for `https`, which the client reports as one), `response`, `status N`, `redirect` (none to follow, or more than 3), `body`, or `timed out after N ms` (the per-image deadline, which a wait for a transfer slot counts toward; timeouts share this code) |
| `image_fetch_overloaded` | `503` | the fetcher was at capacity — 8 transfers in flight and 8 more waiting, process-wide — so this image was refused before any connection. `Retry-After: 1`, `error.type: overloaded_error`, `error.param: null`: the request is fine, retry it after the delay (the admission layer's own overload is the same shape with `code: null`) |

An image whose body is larger than 32 MiB is `image_too_large`, as for a data
URI or a file.

The engine's two are answered **before** the template is rendered, an image is
decoded or the vision tower runs, on both chat endpoints and streamed or not —
as a plain JSON `400`, never a `500` after the encode and never an SSE `error`
event after a `200`:

- `NOT_A_HYBRID_MODEL` — the loaded model is a dense (`qwen3`-family) model;
  image rows need a hybrid `qwen35` (Bonsai 2) model.
- `BACKEND_UNAVAILABLE` — reserved: no shipped executor raises it. It is the
  refusal of a hybrid engine whose decode runs on an executor that cannot
  prefill image rows; both hybrid executors that ship today (Metal and CPU)
  can, so this is a guard kept for a future executor, not something a current
  build answers. The check asks the engine itself, so it stops applying by
  itself when an executor learns to prefill rows. With `--backend auto` a pool
  that could not prefill rows would be rebuilt on the CPU model (one `INFO` line
  says why); an explicit `--backend` is never overridden, and a start-up warning
  says so.

#### Timeouts and image requests

`/v1/chat/completions`, `/v1/chat/completions/extended` and `/v1/completions`
— and, with `--rag`, `/rag/query` — each carry a deadline of their own inside
the handler (the same `--request-timeout-ms`). When it expires the in-flight
generation is cancelled (a request with several choices or prompts starts none
of the rest) and the client gets `504` with `error.code: request_timeout`, a
message naming the stage and `error.phase` set to `preparing`, `image_fetch`
(a remote image was being fetched; see [Remote images](#remote-images)),
`vision_encode`, `waiting_for_engine`, `prefill` or `decode` (with the tokens
generated so far in `error.generated_tokens` when known); a stream that is
already open carries the same error in its terminal SSE `error` event,
followed by `[DONE]`. `/rag/query` is never streamed and reads no images: its
phases are `preparing` (retrieving context and tokenising the prompt),
`waiting_for_engine`, `prefill` and `decode`. A client that disconnects
cancels its generation the same way, streamed or not.
The admission layer's `408` fires two seconds *later*: a backstop for the
other routes (`/v1/embeddings`, `/v1/models`, the probes, `/admin/*`,
`/rag/index`, `/rag/stats`); on the generation routes only a request whose
body takes longer than that to arrive can still get it. `/rag/index` has no
deadline of its own, but a request that is over — its client disconnected, or
the `408` ended it — stops indexing between documents and leaves the index as
it was.
It uses the same error envelope with `error.type: timeout_error` and the same
`error.code: request_timeout`, and no `error.phase` (it cannot know where the
request was). `/health`, `/readyz`, `/metrics` and `/ui/health` are answered
outside the concurrency budget but are bound by this timeout like any other
request.

An image request is seconds on the Metal hybrid runner and tens of seconds to
minutes on the CPU model (figures under [`run`](#run)). With `--mmproj` and
`--request-timeout-ms` below 300000, `serve` warns at start-up and names the flag
to raise; raise it for image workloads, and keep a reverse proxy's read timeout
above it.

#### Model id (`GET /v1/models`)

The listed `id` follows one rule: the GGUF's `general.name` when it is at least 4
characters long and is not, case-insensitively, one of `hf`, `model`, `gguf` or
`llama`; otherwise the model file's stem (`Ternary-Bonsai-2-27B-PQ2_0` for
`Ternary-Bonsai-2-27B-PQ2_0.gguf` — the 27B GGUFs carry `general.name = "Hf"`).
The same id is the `model` member of every completion. `/admin/config` keeps
reporting the model's own `general.name` as `model.id`, placeholder or not.

---

### `info`

Display model metadata parsed from a GGUF file. It prints only values actually
present in the file (a `-` or JSON `null` otherwise), never a fabricated
default. For a Bonsai 2 `qwen35` hybrid it reports the 16 full-attention / 48
Gated-DeltaNet layer split, the resolved weight type with its ggml id (PQ2_0 =
142, PTQ1_0 = 143), the `prism.hadamard.*` contract, the KV and recurrent bytes
per sequence, and the executor `--backend auto` would really decode on.

| Flag | Help |
|------|------|
| `-m, --model <PATH>` | GGUF file; default env `OXI_MODEL`. |
| `--json` | Machine-readable JSON (logging is not initialised for this mode, so the document is the only thing on stdout). |
| `--mmproj <PATH>` | Also report a Bonsai 2 vision projector: which tower `run`/`chat`/`serve --mmproj` would build for the executor `--backend auto` resolves to, what each tower keeps resident, the KV window left with room for the Metal tower, the process's Metal sessions against their ceiling, and whether remote image URLs would be fetched under the environment's settings (`OXI_ALLOW_IMAGE_URL_FETCH`, `OXI_IMAGE_URL_TIMEOUT_MS`, `OXI_IMAGE_URL_ALLOW_HOSTS`: the deadline and the allowlist; `vision.remote_images` in `--json`). Refused for a model that is not `qwen35`. |
| `--image-max-tokens <N>` | Per-image token budget the towers are sized for (default 1024; needs `--mmproj`). |
| `--prefill-chunk <N>` | Report the prefill chunk a `qwen35` engine would run in (on the Metal runner, the call size it would be built with). |

```bash
oxibonsai info --model models/Ternary-Bonsai-1.7B.gguf --json
```

---

### `build-info`

Print what this binary was built with: the enabled Cargo features, the compiled-in
kernel tiers, the kernel tier actually detected at runtime (with the reason), and a
best-effort git commit hash. It takes no model — unlike `info`, it never requires
one — and is the first thing to attach to a bug report.

```bash
oxibonsai build-info
```

---

### `benchmark`

Throughput benchmark. With `--model` it benchmarks the real loaded model end to
end (tokenizer, prefill, decode). Without `--model`, `--synthetic` must be
passed explicitly to run the untrained 2-layer toy benchmark; its numbers are not
representative of any real model and are never the silent default.

| Flag | Default | Help |
|------|---------|------|
| `-m, --model <PATH>` | env `OXI_MODEL` | Real GGUF to benchmark; when given, `--synthetic` is ignored. |
| `--synthetic` | off | The toy benchmark; only meaningful without `--model`. |
| `--tokenizer <PATH>`, `--tokenizer-backend <B>` | as for `run` | For `--model`. |
| `--tokens <N>` | `100` | Tokens generated in the timed pass. |
| `--warmup <N>` | `10` | Warm-up tokens before timing. |
| `--temperature <F>` | `0.7` | Sampling temperature. |
| `--seed <N>` | flag, `[sampling].seed`, else `42` | Seed. |

```bash
oxibonsai benchmark --model models/Ternary-Bonsai-1.7B.gguf --tokens 200 --warmup 20
```

---

### `quantize`

Re-encode a GGUF at another precision. It streams tensor by tensor: each source
tensor is dequantised to f32 and re-encoded through the real
`oxibonsai_model::export` pipeline, so only one tensor is resident at a time, and
the source's architecture and tokenizer metadata are carried into the output.

| Flag | Default | Help |
|------|---------|------|
| `--input <PATH>` | *(required)* | Input GGUF. |
| `--output <PATH>` | *(required)* | Destination file. |
| `--format <FMT>` | `q1_0` | One of `f32`, `q1_0` (Q1_0_g128), `tq2_0_g128` (ternary), `fp8_e4m3`, `fp8_e5m2`, `q4_0`, `q8_0`, `q2_k`, `q3_k`, `q4_k`, `q5_k`, `q6_k`, `q8_k`. Anything else fails fast with no file written. |
| `--force` | off | Skip the up-front memory guard. It estimates whether dequantising the single largest tensor to f32 fits in this machine's available RAM (a conservative 8 GiB when that cannot be determined) and fails fast with a clear message instead of the OS OOM killer. |

```bash
oxibonsai quantize --input models/model-f16.gguf --output models/model-q1_0.gguf --format q1_0
```

---

### `validate`

Validate that a GGUF file is well-formed and print a metadata summary; exits
non-zero when the model configuration cannot be parsed.

```bash
oxibonsai validate --model models/Ternary-Bonsai-1.7B.gguf
```

| Flag | Help |
|------|------|
| `-m, --model <PATH>` | *(required)* GGUF file. |

---

### `convert`

Convert a HuggingFace safetensors model, or an ONNX `MatMulNBits` model, to GGUF.

| Flag | Default | Help |
|------|---------|------|
| `--from <PATH>` | *(required)* | Directory holding `model.safetensors` (or shards) and `config.json`; with `--onnx`, the ONNX model file instead. |
| `--to <PATH>` | *(required)* | Output GGUF path. |
| `--quant <FMT>` | `tq2_0_g128` | `tq2_0_g128` (ternary {-1, 0, +1}, group 128), `q1_0_g128` (1-bit sign + FP16 group scale), `pq2_0` (PrismML 2-bit ternary, ggml id 142), `ptq1_0` (PrismML 1.75-bit ternary, id 143) or `q2_0_g64` (mainline group-64 Q2_0). Anything else is rejected before any work is done. |
| `--onnx` | off | Treat `--from` as an ONNX model (`MatMulNBits`, bits=2) and use the ONNX-to-GGUF converter. |
| `--allow-unmapped` | off | Proceed, with a warning per tensor, when the checkpoint holds tensors the converter cannot map onto a GGUF name; they are dropped from the output. Without it such a checkpoint fails loudly. |

```bash
oxibonsai convert --from ./unpacked-safetensors --to models/Ternary-Bonsai-1.7B.gguf --quant tq2_0_g128
oxibonsai convert --onnx --from path/to/model.onnx --to models/Ternary-Bonsai-1.7B.gguf
```

---

### `eval`

Evaluate a model against a JSONL dataset with the real harness, wired to the
loaded engine end to end. It needs the `eval` build feature (`cargo build
--features eval`). The dataset is loaded and validated **before** the model is
opened, so a malformed file fails fast.

| Flag | Default | Help |
|------|---------|------|
| `-m, --model <PATH>` | env `OXI_MODEL` | GGUF file. |
| `--task <TASK>` | `rouge` | Generation-based: `rouge`, `bleu`, `chrf`, `meteor`, `qa`, `gsm8k`, `exact-match` (dataset `{"input", "expected_output"}`). Teacher-forced: `perplexity` (dataset `{"input"}`). Logit-based: `mmlu`, `arc-easy`, `arc-challenge`, `calibration` (dataset `{"id", "question", "choices", "correct_answer", "subject"?}`), `hellaswag`, `winogrande`, `boolq`, `truthfulqa-mc1`, `truthfulqa-mc2` (each with its own shape; `oxibonsai eval --help` lists them). |
| `--dataset <PATH>` | *(required)* | JSONL dataset; the shape depends on `--task`. |
| `--limit <N>` | all | Only the first N examples. |
| `--max-tokens <N>` | `128` | Tokens generated per example (generation-based tasks only). |
| `--max-seq-len <N>` | `4096`, or `[model].max_seq_len` | Context window. |
| `--tokenizer <PATH>`, `--tokenizer-backend <B>`, `--allow-vocab-mismatch` | as for `run` | Tokenizer selection and the TOK-08 check. |
| `--report-json <PATH>`, `--report-markdown <PATH>` | — | Also write the report in that format. |

```bash
oxibonsai eval --model models/Ternary-Bonsai-1.7B.gguf --dataset eval_data.jsonl \
  --task rouge --max-tokens 128 --report-json report.json --report-markdown report.md
```

---

### `tokenizer`

Manage the Qwen3 tokenizer. Two subcommands: `download` and `info`.

```text
oxibonsai tokenizer <download|info> [OPTIONS]
```

| Subcommand | Flag | Default | Help |
|------------|------|---------|------|
| `download` | `--output <PATH>` | `models/tokenizer.json` | Destination; parent directories are created. |
| `download` | `--repo <REPO>` | `Qwen/Qwen3-8B` | HuggingFace repository holding a `tokenizer.json`. |
| `download` | `--force` | off | Overwrite an existing file without prompting. |
| `info` | `--path <PATH>` | `models/tokenizer.json` | Show the vocabulary size and model type stored in a `tokenizer.json`. |

---

### `pull`

Download a Bonsai 2 / Bonsai model artifact, or any `http(s)` URL, in Pure Rust
(the outbound client is `oxihttp` over `oxitls`; it trusts the public WebPKI
roots). `pull` never runs implicitly: no model artifact downloads at inference
time. (The one thing fetched at inference time is an `http(s)` image reference,
and only after the operator opts in — see [Remote images](#remote-images).)

```bash
oxibonsai pull bonsai2-27b --vision            # PQ2_0 band + the vision projector
oxibonsai pull bonsai2-27b --band ptq1         # the 1.75-bit band
oxibonsai pull bonsai-8b --out models
```

`<MODEL_OR_URL>` is a known name — `bonsai2-27b`, `bonsai2-27b-ptq1_0`,
`bonsai2-27b-pq2_0`, `bonsai2-27b-mmproj`, `bonsai2-27b-q2_0`, `bonsai-27b-q1_0`,
`ternary-bonsai-27b-pq2_0`, `ternary-bonsai-27b-q2_0` or `bonsai-8b` — or a full
`http(s)://` URL.

| Flag | Default | Help |
|------|---------|------|
| `--out <DIR>` | `models` | Output directory. |
| `--band <BAND>` | `pq2` | For `bonsai2-27b`: `pq2` or `ptq1`. |
| `--vision` | off | For `bonsai2-27b`: also fetch the mmproj vision projector. |
| `--force` | off | Overwrite an existing file; without it `pull` refuses. |

A named entry is verified **fail-closed**: first the GGUF structure (magic, the
expected `general.architecture` and, for a Bonsai 2 language file,
`prism.hadamard.version == 1`), then the exact byte size, then the SHA-256 —
against digests compiled into the binary from the upstream HuggingFace LFS
objects, never a file relative to the working directory, so an installed binary
verifies exactly like a checkout. A mismatch refuses the file and deletes the
partial download. `scripts/checksums.sha256` or `OXIBONSAI_CHECKSUMS_FILE`, when
readable, is an additional cross-check and the only hash authority for a bare
URL. Downloads stream to `<file>.part` with a live progress line and resume from
it (an HTTP `Range` request; a reply that does not start at the resume offset
restarts from byte 0 rather than splicing the wrong bytes). `OXIBONSAI_HF_BASE_URL`
points the whole manifest at a mirror; `OXI_BONSAI2_REPO` and
`OXI_BONSAI2_DEV_REPO` override the two Bonsai 2 repositories.

---

## `oxibonsai-serve` — standalone server

A separate binary (crate `oxibonsai-serve`) running the same OpenAI-compatible
router with a layered configuration. It parses argv by hand (no `clap`) and
merges four layers, lowest first:

```text
built-in defaults  <  TOML file (--config)  <  OXIBONSAI_* environment  <  CLI flags
```

Unlike the `oxibonsai` binary, **the environment overrides the TOML file here**.
A flag counts as explicit whenever it is literally present in argv — even at the
value of its default — so `--port 8080` does override a different port set in
the TOML file or the environment. Unlike `oxibonsai serve`, the standalone
binary reads no GGUF sampling defaults: its `--temperature`, `--max-tokens` and
`[sampling]` values are the server's own defaults for a request that sets none.

```text
oxibonsai-serve [OPTIONS]
```

| Flag | Default | Help |
|------|---------|------|
| `--config <PATH>` | — | TOML configuration file (optional). |
| `--host <HOST>` | `127.0.0.1` | Bind host. A non-loopback host requires `--bearer-token`, `--bearer-token-file` or `--insecure-no-auth`, otherwise start-up is refused. |
| `--port <PORT>` | `8080` | Bind port. |
| `--model <PATH>` | — | GGUF model file. |
| `--tokenizer <PATH>` | — | Tokenizer file or directory (optional). |
| `--max-tokens <N>` | `256` | Default `max_tokens` for a request that sets none. |
| `--temperature <F>` | `0.7` | Default temperature. |
| `--seed <N>` | `42` | RNG seed. |
| `--log-level <LEVEL>` | `info` | `error`, `warn`, `info`, `debug`, `trace` or `off`; anything else is rejected at start-up. `RUST_LOG` is not consulted. |
| `--bearer-token <TOKEN>` | — | Bearer token for protected endpoints (at least 16 bytes; visible through `ps`). |
| `--bearer-token-file <PATH>` | — | Read the token from this file instead (`ps`-safe); takes precedence over `--bearer-token`. |
| `--admin-token <TOKEN>` | `OXI_ADMIN_TOKEN` | Credential for `/admin/*`; `403` while neither is set. |
| `--insecure-no-auth` | off | Allow a non-loopback `--host` with no bearer token configured. An admin token alone does not satisfy the check: it protects `/admin/*` but leaves the inference endpoints open. |
| `--cors-origin <ORIGIN>` | none = **no CORS headers** | Allowed origin; repeatable. |
| `--cors-allow-credentials` | off | Send `Access-Control-Allow-Credentials: true`; rejected together with a literal `*` origin. |
| `--rate-limit-rpm <F>` | disabled | Per-client requests per minute. |
| `--rate-limit-burst <F>` | `20` | Per-client burst capacity. |
| `--max-body-bytes <N>` | `4194304` | Request-body ceiling (`413` above it). |
| `--enable-ui` | off | Mount the chat UI at `GET /ui`. |
| `--max-output-tokens <N>` | `8192` | Hard ceiling on a request's `max_tokens`. |
| `-h, --help`, `-V, --version` | — | Print help or version and exit. |

Some knobs have no flag and are reachable only through TOML or the environment:
`top_p`, `max_input_tokens`, `max_concurrent_requests`, `engine_pool_size`,
`per_request_timeout_ms`, `metrics_enabled`, `metrics_path`, `tokenizer_kind`
and `quantization_hint`. The model's checksum is verified at start-up exactly as
for `oxibonsai serve` (`OXIBONSAI_CHECKSUMS_FILE`, default
`scripts/checksums.sha256` relative to the working directory). Logs go to
stdout, and the effective admission ceiling is `min(max_concurrent_requests,
4 x pool size)` (`/health`, `/readyz`, `/metrics`, `/metrics/serve`, the
`metrics_path` alias and `/ui/health` are answered outside it). The tokenizer is
resolved as for `run`: an explicit path
(hard-checked against the model's vocabulary), else a compatible
`tokenizer.json` next to the model, else the vocabulary and chat template
embedded in the GGUF (every Bonsai 2 file carries one).

Start-up validation refuses, naming the field: port `0`; `default_max_tokens`
outside `1..=8192`; a temperature outside `0.0..=2.0` or a top-p outside
`0.0..=1.0`; a `log_level` outside the six names above; a `model.path` or
`tokenizer.path` that does not exist; a `tokenizer.kind` other than
`huggingface` or `hf`; a bearer or admin token shorter than 16 bytes; a zero
limit; a rate or burst that is not a positive number; a wildcard CORS origin
combined with credentials; and a `metrics_path` that is empty, does not start
with `/`, or collides with a built-in route.

TOML sections are `[bind]`, `[model]`, `[tokenizer]`, `[sampling]`, `[limits]`,
`[auth]`, `[observability]`, `[cors]`, `[rate_limit]`, `[ui]`, plus a top-level
`seed`. The keys are exactly these; an unknown key is **ignored without an
error**, so check spelling against the list:

```toml
seed = 42                          # top-level keys must precede the first [table]

[bind]
host = "127.0.0.1"
port = 8080

[model]
path = "models/Ternary-Bonsai-1.7B.gguf"
quantization_hint = "TQ2"          # log/metrics label only

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
per_request_timeout_ms = 60_000
engine_pool_size = 2               # omit for the sizing rule in the serve section
max_body_bytes = 4_194_304
max_output_tokens = 8192

[auth]
bearer_token_file = "/etc/oxibonsai/token"   # at least 16 characters
# admin_token = "..."              # else OXI_ADMIN_TOKEN; /admin/* is 403 without one
# insecure_no_auth = true          # only to expose a non-loopback host with no token

[cors]
allowed_origins = ["https://app.example.com"]
allow_credentials = false

[rate_limit]
rpm = 120
burst = 20

[ui]
enabled = false

[observability]
log_level = "info"
metrics_enabled = true
metrics_path = "/metrics"
```

`crates/oxibonsai-serve/examples/server_config.toml` is the fully commented
starting point; a test parses and validates it, so it stays in step with the
schema.

---

## HTTP surface shared by both servers

| Method and path | Auth | Notes |
|-----------------|------|-------|
| `POST /v1/chat/completions` | bearer | Streaming SSE, tool calls (streamed too), `reasoning_content` for `<think>` blocks, `enable_thinking` / `reasoning_effort`, per-request `seed`, `min_p`, penalties and `logprobs`, and `image_url` parts with a projector. A non-empty `logit_bias` is refused with `400` (there is no per-token bias seam in the sampler). |
| `POST /v1/chat/completions/extended` | bearer | The same engine with this server's extension fields. |
| `POST /v1/completions` | bearer | Batched prompts, real SSE streaming, `echo`, `stop`, `seed`, `logprobs` in the legacy shape. Refused with `400` naming the field: a non-empty `suffix`, `n` other than 1, `best_of` other than 1, a non-empty `logit_bias`. Any undeclared field is `400 unknown_parameter`. |
| `POST /v1/embeddings` | bearer | Per `--embedding-backend`; `501` when none is available. |
| `GET /v1/models`, `GET /v1/models/{model}` | bearer | `max_context_length` is the window in force. An API route: inside the concurrency budget, so `503` + `Retry-After` while the server is at its limit. |
| `GET /health` | **none** | Liveness: `200` once the router is mounted. Outside the concurrency budget: it answers while the server is at its limit. |
| `GET /readyz` | bearer | Readiness: `200` with `{"status":"ready","model_loaded":true,"engine_slot_available":...}` while a model is loaded and the engine pool can serve, and `503` `not_ready` only for a server that cannot serve at all. A saturated server is ready: with every replica busy and the concurrency budget full it still answers `200`, with `engine_slot_available: false` (whether a replica is idle right now; information, not part of the verdict). It is *not* exempt from bearer auth. Outside the concurrency budget, so it never answers the admission layer's `overloaded_error`. |
| `GET /metrics` | **none** | Prometheus text exposition. Outside the concurrency budget. |
| `GET /metrics/serve` | **none** | Standalone binary only: the server's own request counters and duration histogram. Outside the concurrency budget, as is the `observability.metrics_path` alias. |
| `/admin/status`, `/admin/config`, `/admin/cache-stats`, `/admin/workload-stats`, `POST /admin/reset-metrics` | admin token | `403` while no admin token is configured; the bearer token does not unlock it. Inside the concurrency budget. |
| `/rag/index`, `/rag/query`, `/rag/stats` | bearer | `oxibonsai serve --rag` only (needs the `rag` feature). `/rag/query` carries the generation routes' deadline and is cancelled when its client disconnects; `/rag/index` is cancelled on disconnect too, between documents, and leaves the index unchanged. |
| `GET /ui`, `GET /ui/health` | bearer; `/ui/health` none | Only with `--enable-ui`. `/ui/health`, the page's own liveness ping, is outside the concurrency budget; `/ui` is inside it. |

The middleware order, innermost first, is routes, admission (concurrency limit
and timeout), body limit and `Content-Length` precheck, rate limit, bearer auth,
CORS. A rate-limited or unauthenticated request therefore never occupies a
concurrency permit, and a CORS preflight is answered before auth runs. Admission
wraps every route: only `/health`, `/readyz`, `/metrics` and `/ui/health` (and,
on the standalone binary, `/metrics/serve` and the metrics alias) are answered
outside its concurrency budget, so they keep answering while the server is at its
limit, and a route added later is inside the budget unless it is named in the
exempt list. The probe routes are still bound by the body limit, the rate limit
(which exempts only `/health` and `/metrics`), bearer auth, CORS and the request
timeout, exactly like every other route. The rate
limiter keys on the connecting peer's address: neither binary exposes a
trusted-proxy list, so behind a reverse proxy every client shares the proxy's
bucket. Rate-limit at the proxy in that topology (see the
[deployment guide](DEPLOYMENT.md)). Chat requests have control tokens
(`<|im_start|>` and the like) stripped from client-supplied message text before
the template is assembled, so a client cannot forge turn boundaries;
`OXI_DISABLE_PROMPT_SANITIZATION=1` turns that off and should stay unset on any
server a third party can reach.

### Responses from the server's own limits

The layers around the handlers answer with the API's error envelope
(`{"error": {"message", "type", "param", "code"}}`) unless noted:

| Status | `error.type` | `error.code` | From | When |
|--------|--------------|--------------|------|------|
| `503` | `overloaded_error` | `null` | admission | The concurrency budget is full. Carries `Retry-After: 1`. Never sent for `/health`, `/readyz`, `/metrics` or `/ui/health`. |
| `408` | `timeout_error` | `request_timeout` | admission | A request on a route with no deadline of its own (`/v1/embeddings`, `/v1/models`, the probes, `/admin/*`, `/rag/index`, `/rag/stats`) — or a generation request whose body was still arriving — outlasted `--request-timeout-ms` plus two seconds. No `error.phase`. |
| `504` | `server_error` | `request_timeout` | `/v1/chat/completions`, `/v1/chat/completions/extended`, `/v1/completions`, `/rag/query` (with `--rag`) | The handler's own deadline (`--request-timeout-ms`) expired and the generation was cancelled; `error.phase` names the stage. |
| `429` | `rate_limit_error` | none | rate limiter | The client's bucket is empty; `Retry-After` and `retry_after_ms`. `/health` and `/metrics` are never limited. |
| `401` | `auth_error` | `null` | bearer auth | A missing or wrong token. |
| `413` | `invalid_request_error` | `content_too_large` | body limit | A declared `Content-Length` above `--max-body-bytes`, refused before the request reaches admission (a body found to exceed it only while being read is also a `413`, from the JSON extractor). |

Both timeouts carry the same `error.code`, `request_timeout`, so a client can
handle "the request outlasted the server's timeout" by code; the `408` stays the
status of the admission backstop.

---

## Environment variables

There are two prefixes. **`OXI_*`** names model paths and feature toggles read
by the `oxibonsai` binary and the libraries beneath it; **`OXIBONSAI_*`**
names server configuration and runtime knobs, read by both servers (and by the
kernels and runtime). Variables that exist only to drive tests and release-gate
legs (`OXI_BONSAI2_*_GGUF`, `OXI_REQUIRE_MODEL_FILES`, `OXIBONSAI_CAPABILITY_REPORT`,
`OXIBONSAI_M08_RUN_LONG` and the like) are documented in the scripts that use
them, not here.

### Model and tokenizer paths (`oxibonsai`)

| Variable | Used by | Controls |
|----------|---------|----------|
| `OXI_MODEL` | `run`, `chat`, `serve`, `info`, `benchmark`, `eval` | GGUF path when neither `--model` nor `[model].model_path` is given. |
| `OXI_TOKENIZER` | `run`, `chat`, `serve`, `eval` | `tokenizer.json` path or directory; otherwise auto-detected next to the model. |
| `OXI_DIT_GGUF` | `image`, `repl` | DiT GGUF; otherwise the data-directory fallback described under [`image`](#image). |
| `OXI_VAE_WEIGHTS` | `image`, `repl` | VAE weights: a `.safetensors` file or a `.npy` directory; same fallback. |
| `OXI_TE_4BIT` | `image`, `repl` | 2.1 GB 4-bit MLX text-encoder `model.safetensors`; takes precedence over `OXI_TE_WEIGHTS`. |
| `OXI_TE_WEIGHTS` | `image`, `repl` | Fallback text encoder as an f32 `.npy` directory (used only when `--te` and `OXI_TE_4BIT` are both unset). |
| `OXI_TE_TOKENIZER_DIR` | `image`, `repl` | Text-encoder tokenizer directory; default the TE directory. |
| `OXI_ALLOW_IMAGE_URL_FETCH` | `run`, `chat`, `serve` | Fallback for `--allow-image-url-fetch` (`1`, `true`, `yes` or `on`, case-insensitive): fetch remote image references under the address policy ([Remote images](#remote-images)). The flag wins even when this is `0`. Read only by a command given `--mmproj`: a text-only command never consults it. |
| `OXI_IMAGE_URL_TIMEOUT_MS` | `run`, `chat`, `serve` | Fallback for `--image-url-timeout-ms`, the per-image fetch deadline (default 10000). Read only by a command given `--mmproj`, and only when remote images are enabled; there, `0` or a non-number is a start-up error naming the variable. A text-only command never consults it, so a stale value cannot stop it. |
| `OXI_IMAGE_URL_ALLOW_HOSTS` | `run`, `chat`, `serve` | Comma-separated `host[:port]` entries exempt from the public-address rule; `--image-url-allow-host` entries are added to them. Read only by a command given `--mmproj`; there, a malformed entry is a start-up error naming it, and so is the variable set without the opt-in. A text-only command never consults it, so a stale value cannot stop it. |
| `OXI_MEDIA_PATH` | `serve` | Fallback for `--media-path`. A server without a projector never consults it, so a stale value cannot stop it. |

Resolution for the text encoder is `--te`, then `OXI_TE_4BIT`, then
`OXI_TE_WEIGHTS`, then an existing `models/te` in the data directory.

### Model-format and `pull` overrides

| Variable | Controls |
|----------|----------|
| `OXI_FORCE_Q2_LAYOUT` | Operator override for the three on-disk layouts that share ggml type id 42. Read by every loader, case-insensitively: `d-first` (also `dfirst`, `pq2`), `qs-first` (also `qsfirst`, `tq2`, `legacy`) or `g64` (also `q2_0_g64`). A typo is a hard error naming the accepted values — it never degrades to the automatic sniff. See [`docs/models.md`](models.md). |
| `OXIBONSAI_HF_BASE_URL` | Point `pull` at a mirror (or a local test server). |
| `OXI_BONSAI2_REPO`, `OXI_BONSAI2_DEV_REPO` | Override the two Bonsai 2 HuggingFace repositories (`pull` and `scripts/download_ternary.sh`). |
| `OXIBONSAI_CHECKSUMS_FILE` | Checksum manifest for `serve`, `oxibonsai-serve` and `pull`; default `scripts/checksums.sha256` relative to the working directory. |

### Server configuration

Read by `oxibonsai serve` where a flag exists, and by `oxibonsai-serve` as its
environment layer. On the standalone server the variables mapped in `env.rs`
(host, port, paths, limits, sampling defaults, log and metrics settings, seed,
UI and output ceiling) fail start-up on a malformed value, while the hardening
overrides (`OXIBONSAI_ADMIN_TOKEN`, `OXIBONSAI_INSECURE_NO_AUTH`,
`OXIBONSAI_CORS_*`, `OXIBONSAI_RATE_LIMIT_*`, `OXIBONSAI_MAX_BODY_BYTES`)
silently ignore a value they cannot parse, so confirm them in the start-up log.
An unrecognised `OXIBONSAI_*` variable is ignored.

| Variable | Controls | Default |
|----------|----------|---------|
| `OXIBONSAI_HOST`, `OXIBONSAI_PORT` | Bind address (standalone). | `127.0.0.1`, `8080` |
| `OXIBONSAI_MODEL_PATH` | GGUF path (standalone). | — |
| `OXIBONSAI_TOKENIZER_PATH`, `OXIBONSAI_TOKENIZER_KIND` | Tokenizer path and kind (`huggingface` or `hf`; anything else is rejected; standalone). | — |
| `OXIBONSAI_MAX_TOKENS`, `OXIBONSAI_TEMPERATURE`, `OXIBONSAI_TOP_P` | Request sampling defaults (standalone). | `256`, `0.7`, `1.0` |
| `OXIBONSAI_MAX_INPUT_TOKENS` | Maximum prompt length in tokens (standalone). | `8192` |
| `OXIBONSAI_MAX_CONCURRENT` | Admission ceiling before the `4 x pool size` cap (standalone). | `32` |
| `OXIBONSAI_REQUEST_TIMEOUT_MS` | Per-request timeout (standalone). | `60000` |
| `OXIBONSAI_ENGINE_POOL_SIZE` | Engine replicas (both servers); see the pool-size table. | host-dependent |
| `OXIBONSAI_MAX_OUTPUT_TOKENS` | Hard `max_tokens` ceiling (standalone). | `8192` |
| `OXIBONSAI_MAX_BODY_BYTES` | Request-body ceiling (both). | 4 MiB |
| `OXIBONSAI_BEARER_TOKEN` | Bearer token (both); at least 16 bytes. | — |
| `OXIBONSAI_BEARER_TOKEN_FILE` | Token file (`oxibonsai serve` only; the standalone uses `--bearer-token-file` or `[auth].bearer_token_file`). | — |
| `OXIBONSAI_ADMIN_TOKEN`, `OXI_ADMIN_TOKEN` | `/admin/*` credential, in that order of precedence (both). | unset = `403` |
| `OXIBONSAI_INSECURE_NO_AUTH` | Acknowledge a non-loopback bind with no token (both). | off |
| `OXIBONSAI_CORS_ORIGIN` | Comma-separated allowed origins (both). | none |
| `OXIBONSAI_CORS_ALLOW_CREDENTIALS` | `Access-Control-Allow-Credentials` (both). | off |
| `OXIBONSAI_RATE_LIMIT_RPM`, `OXIBONSAI_RATE_LIMIT_BURST` | Per-client rate limit (both). | disabled, `20` |
| `OXIBONSAI_ENABLE_UI` | Mount `/ui` (standalone). | off |
| `OXIBONSAI_LOG_LEVEL` | Log filter (standalone). | `info` |
| `OXIBONSAI_METRICS_ENABLED`, `OXIBONSAI_METRICS_PATH` | Gate `/metrics`, and serve it at an extra path (standalone). Booleans accept `true`/`yes`/`on`/`1` and `false`/`no`/`off`/`0`. | `true`, `/metrics` |
| `OXIBONSAI_SEED` | Sampler seed. Standalone: the environment layer, above the TOML and below `--seed`. `oxibonsai serve` has no `--seed` and consults it first, then `[sampling].seed`. | `42` standalone; pseudo-random for `oxibonsai serve` |
| `OXIBONSAI_QUANTIZATION_HINT` | Quantisation label (standalone). | — |
| `OXIBONSAI_CUDA_DEVICE` | CUDA device ordinal; set by `serve --cuda-device`. | `0` |

### Kernel, runtime and diagnostics knobs

| Variable | Controls |
|----------|----------|
| `OXIBONSAI_KERNEL_TIER` | Opt-in INT8 dot-product tier for the CPU kernels: `int8-scalar`, `neon-int8`, `neon-dot`, `neon-i8mm` or `avx512-vnni`. It quantises the activation to INT8 once per call and accumulates exactly in `i32`; it is lossy and **never selected by default** (unset, the f32 kernels run bit-identically). It does not divert the GPU tier while a GPU prefill runs, but a declined or failed GPU prefill, and the batched CPU embedding pass on any engine, run on it. A named tier the CPU lacks is clamped with a warning; an unknown name selects nothing. |
| `OXIBONSAI_METAL_MAX_SESSIONS` | The ceiling a Metal engine pool is sized against (default 4; `0` or a non-number keeps the default). It bounds the replica count a pool is built with; it is **not** enforced when a session opens, so a Metal vision tower (a second session) or any other session counts against memory but not against it. `oxibonsai info --mmproj` reports the process's Metal sessions against it. |
| `OXIBONSAI_SPEC` | `1` opts in to n-gram speculative decoding on the greedy GPU path; default off, because the batch-shaped verify kernels are not proven bit-identical to the single-token path and could, in principle, change a token at a near-tie. |
| `OXIBONSAI_FORCE_CPU_DECODE_AFTER` | A test hook: forces the rest of a GPU greedy decode onto the CPU fallback after this many tokens, to exercise the Metal-to-CPU KV-cache rebuild without a real dispatch failure. Do not set it in production. |
| `OXIBONSAI_FORCE_METAL_TAIL_FAIL` | A fault-injection hook read by the production dense Metal forward (any value, read once per process): the ternary fused forward fails at its final-norm → LM-head tail after every layer's weights are already on the GPU, reproducing a deterministic tail failure without a broken GPU, to exercise the fallback that must not upload the model a second time. Do not set it in production. |
| `OXIBONSAI_PROFILE_GPU` | When set (any value; Metal builds), print a GPU timing summary after generation. |
| `OXIBONSAI_PROFILE` | When set (any value; Metal builds), full per-layer profiling output. |
| `OXI_DISABLE_PROMPT_SANITIZATION` | `1`, `true`, `yes` or `on` disables the chat-prompt control-token stripping. Leave unset. |

### Image-pipeline toggles

| Variable | Controls |
|----------|----------|
| `OXI_DIT_GPU`, `OXI_DIT_ATTN_GPU`, `OXI_VAE_GPU` | The GPU DiT matmuls, DiT flash attention and VAE decode. **On by default** in a GPU-featured build; `=0` opts out of that stage and the CPU fallback is always available. |
| `OXI_DIT_FUSED` | CUDA only: the fused DiT path; on by default, `=0` opts out. |
| `OXI_TE_GPU` | The GPU text-encoder GEMM. Off for `image` (set `=1` to enable); `repl` sets it unless `--cpu-te`. |
| `OXI_TE_GEMM_F32` | `=1` forces the f32 text-encoder GEMM instead of the default bf16 one where the device supports it. |
| `OXI_TE_RESIDENT` | `=1` pins the dequantised f32 text encoder (about 16 GB) in memory; `=0` forces it transient. Default is source-aware (see [`repl`](#repl)). |
| `OXI_VAE_NO_IMPLICIT_CONV` | `=1` forces the legacy tiled-im2col VAE convolution on Metal (a diagnostic). |
| `OXI_IMAGE_TIMING` | When set (any value), print per-stage timings. |
