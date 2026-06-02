# OxiBonsai Image Generation Guide

End-to-end walkthrough for `oxibonsai image` — text-to-image generation with the
**Bonsai-Image** model (a FLUX.2-Klein-based ternary diffusion model), running
entirely in Pure Rust. The pipeline is **Text Encoder (Qwen3-4B, 4-bit) → DiT
(FLUX.2-Klein ternary, TQ2) → VAE (`AutoencoderKLFlux2`) → PNG**, GPU-accelerated
(Metal is default-on on Apple Silicon; a CUDA backend covers NVIDIA), with a CPU
fallback. Every stage is parity-validated against the MLX reference at cosine
similarity ≥ 0.999.

This guide takes you from nothing (no weights on disk) to a generated PNG. For
the exhaustive flag/env reference across every subcommand, see
[`docs/CLI.md`](CLI.md); to return to the project overview, see
[`README.md`](../README.md).

## Contents

- [What `oxibonsai image` is](#what-oxibonsai-image-is)
- [Asset acquisition (from nothing to an image)](#asset-acquisition-from-nothing-to-an-image)
  - [1. DiT (the ternary transformer)](#1-dit-the-ternary-transformer)
  - [2. Text encoder (Qwen3-4B, 4-bit)](#2-text-encoder-qwen3-4b-4-bit)
  - [3. VAE decoder (AutoencoderKLFlux2)](#3-vae-decoder-autoencoderklflux2)
  - [4. Tokenizer](#4-tokenizer)
- [Environment / `.env` setup](#environment--env-setup)
- [`oxibonsai image` flag reference](#oxibonsai-image-flag-reference)
- [Generating an image](#generating-an-image)
- [Performance](#performance)
- [Validation / parity](#validation--parity)
- [Provenance](#provenance)

---

## What `oxibonsai image` is

`oxibonsai image` generates a PNG (512×512 by default) from a text prompt using
the Bonsai-Image model. The entire pipeline is Pure Rust and GPU-accelerated —
Metal is **default-on** on Apple Silicon, and a CUDA backend is available on
NVIDIA hardware — with a CPU fallback on every stage:

```text
prompt ──▶ Text Encoder ──▶ DiT ──▶ VAE decoder ──▶ PNG
           (Qwen3-4B,        (ternary FLUX.2    (Autoencoder    (oxiarc-deflate)
            4-bit)            transformer,        KLFlux2)
                              TQ2_0_g128)
```

| Stage | What it is | On-disk format | Selected by |
| --- | --- | --- | --- |
| Text Encoder | Qwen3-4B, 4-bit | MLX 4-bit `.safetensors` | `--te` / `OXI_TE_4BIT` |
| DiT | Ternary FLUX.2-Klein transformer | GGUF, `TQ2_0_g128` | `--dit` / `OXI_DIT_GGUF` |
| VAE decoder | `AutoencoderKLFlux2` | FLUX.2 `.safetensors` (or legacy `.npy` dir) | `--vae` / `OXI_VAE_WEIGHTS` |
| Tokenizer | Qwen3 BPE | `tokenizer.json` | `--tokenizer` / `OXI_TE_TOKENIZER_DIR` |
| PNG encode | `oxiarc-deflate` | PNG | (always) |

---

## Asset acquisition (from nothing to an image)

You need **three model assets** plus a tokenizer. The downloads use the
HuggingFace CLI (`hf`):

```bash
pip install huggingface_hub      # provides the `hf` CLI
```

> The `hf` download is the **only external (non-Rust) step**, and it is used
> **only for downloading** the checkpoints. All conversion and all inference are
> Pure Rust — no Python at runtime.

| Asset | Source repo | File(s) | Convert? | Points at |
| --- | --- | --- | --- | --- |
| DiT | `prism-ml/bonsai-image-ternary-4B-mlx-2bit` | `diffusion_pytorch_model.safetensors` | yes → GGUF (`mlx_image_convert`) | `--dit` / `OXI_DIT_GGUF` |
| Text encoder | `prism-ml/text_encoder-mlx-4bit` | `model.safetensors`, `tokenizer.json` | no (native 4-bit loader) | `--te` / `OXI_TE_4BIT` |
| VAE | `black-forest-labs/FLUX.2-dev` | `vae/diffusion_pytorch_model.safetensors` | no (native safetensors loader) | `--vae` / `OXI_VAE_WEIGHTS` |
| Tokenizer | (ships in the TE repo) | `tokenizer.json` | no | `--tokenizer` / `OXI_TE_TOKENIZER_DIR` |

### 1. DiT (the ternary transformer)

Download the PrismML ternary checkpoint, then convert it to GGUF. The conversion
is Pure Rust (it quantizes the linears to `TQ2_0_g128` and passes the
skip-pattern tensors through as BF16):

```bash
# Download the MLX 2-bit ternary DiT
hf download prism-ml/bonsai-image-ternary-4B-mlx-2bit \
    diffusion_pytorch_model.safetensors --local-dir ./bonsai-dit

# Convert safetensors → GGUF (Pure Rust). The trailing quant arg is optional and
# defaults to tq2_0_g128 (the only format supported today).
cargo run -p oxibonsai-model --example mlx_image_convert --release -- \
    ./bonsai-dit/diffusion_pytorch_model.safetensors ./bonsai-dit.gguf tq2_0_g128
```

The example's argument order is `<model.safetensors> <output.gguf> [quant]`.
Point `--dit` / `OXI_DIT_GGUF` at the resulting `./bonsai-dit.gguf`.

### 2. Text encoder (Qwen3-4B, 4-bit)

Download the 4-bit MLX text encoder. **No conversion** — the native Rust loader
reads the MLX `.safetensors` directly:

```bash
hf download prism-ml/text_encoder-mlx-4bit \
    model.safetensors tokenizer.json --local-dir ./bonsai-te
```

- Point `--te` / `OXI_TE_4BIT` at `./bonsai-te/model.safetensors`.
- Point `--tokenizer` / `OXI_TE_TOKENIZER_DIR` at its **directory**
  (`./bonsai-te`). When `--te` is a `.safetensors` file, the tokenizer dir
  auto-defaults to that file's parent, so you can usually omit it.

A `--te` path ending in `.safetensors` selects the native 4-bit MLX loader;
anything else is treated as a legacy f32 `.npy` directory.

> **Footprint:** the 4-bit text encoder is ≈ 2.1 GB on disk — *not* the old
> 15 GB f32 `.npy` dump. The Rust loader dequantizes on the fly.

### 3. VAE decoder (AutoencoderKLFlux2)

Download the standard FLUX.2 VAE. **No conversion needed** — the native
Pure-Rust safetensors loader reads the diffusers checkpoint directly. This is the
new path added this release
([`crates/oxibonsai-image/src/vae/safetensors.rs`](../crates/oxibonsai-image/src/vae/safetensors.rs)):
it resolves every weight key the decoder asks for, applies the layout/name
transforms in-engine (bf16→f32 lossless decode, conv-weight transpose
`[O,I,kH,kW] → [O,kH,kW,I]`, and the `to_out.0` ModuleList un-nesting), and feeds
the decoder f32 values byte-identical to the old `.npy` path.

```bash
hf download black-forest-labs/FLUX.2-dev \
    vae/diffusion_pytorch_model.safetensors --local-dir ./flux2
```

Point `--vae` / `OXI_VAE_WEIGHTS` at
`./flux2/vae/diffusion_pytorch_model.safetensors`.

> **No more Python export.** The previous workflow required a dev-time Python
> `.npy` export of the VAE weights (`/tmp/bonsai_vae_export_weights.py`); that
> step is now eliminated — the engine reads the `.safetensors` checkpoint
> directly.
>
> **Which VAE source?** The canonical file is
> `black-forest-labs/FLUX.2-dev` → `vae/diffusion_pytorch_model.safetensors`
> (`_class_name = "AutoencoderKLFlux2"`), as documented in the loader. If the
> Bonsai-Image checkpoint you are using ships its **own** `vae/` subfolder
> (e.g. a `bonsai-image-*-mlx` repo bundling `vae/diffusion_pytorch_model.safetensors`),
> that matching file is the equivalent source and can be used directly.
>
> `--vae` also still accepts a **directory of per-tensor `.npy` tensors** for the
> legacy path. The loader auto-detects: a `.safetensors` *file* selects the
> native loader; a *directory* selects the `.npy` reader.

### 4. Tokenizer

The Qwen3 tokenizer (`tokenizer.json`) ships inside the text-encoder repo
(`prism-ml/text_encoder-mlx-4bit`), so the `hf download` in step 2 already fetches
it. Point `--tokenizer` / `OXI_TE_TOKENIZER_DIR` at the directory that contains
it; when `--te` is a `.safetensors` file this defaults to that file's parent, so
it is usually omitted.

If you need to fetch a tokenizer separately, `oxibonsai tokenizer download` pulls
`tokenizer.json` from a HuggingFace repo (defaults: repo `Qwen/Qwen3-8B`, output
`models/tokenizer.json`):

```bash
oxibonsai tokenizer download --output ./bonsai-te/tokenizer.json
```

---

## Environment / `.env` setup

`oxibonsai` auto-loads a `.env` file from the current directory (or any parent),
so you can set the asset paths once. Create a `.env`:

```dotenv
# DiT ternary transformer (GGUF produced by mlx_image_convert)
OXI_DIT_GGUF=./bonsai-dit.gguf

# Text encoder: 4-bit MLX model.safetensors
OXI_TE_4BIT=./bonsai-te/model.safetensors

# Tokenizer directory containing tokenizer.json
# (omit to default to the TE .safetensors' parent dir)
OXI_TE_TOKENIZER_DIR=./bonsai-te

# VAE decoder weights (.safetensors file, or a legacy .npy directory)
OXI_VAE_WEIGHTS=./flux2/vae/diffusion_pytorch_model.safetensors
```

**Precedence** (highest wins): `--flag` > shell environment variable > `.env`
file > built-in default. See [`docs/CLI.md`](CLI.md#environment-variables) for
the full environment-variable table.

**GPU is default-on.** On Apple Silicon, a `metal`-feature build uses the GPU
automatically; on NVIDIA, build with the CUDA feature (below). Per-stage GPU
toggles (e.g. `OXI_VAE_GPU`, `OXI_DIT_ATTN_GPU`) default to on and accept an
opt-out of `"0"`; the CPU fallback is always available. See
[`docs/CLI.md`](CLI.md#environment-variables) for the full toggle list.

---

## `oxibonsai image` flag reference

All flags and defaults are taken directly from `src/main.rs`. For the canonical,
exhaustive table (and every other subcommand) see
[`docs/CLI.md`](CLI.md#image).

| Flag | Short | Default | Env fallback (when flag omitted) | Description |
| --- | --- | --- | --- | --- |
| `--prompt` | `-p` | *(required)* | — | Text prompt. Use `-` to read from stdin. |
| `--out` | `-o` | *(required)* | — | Output PNG path. |
| `--seed` | | `42` | — | RNG seed for the initial noise. |
| `--steps` | | `4` | — | Number of Euler (flow-matching) sampler steps. |
| `--width` | | `512` | — | Image width in pixels. |
| `--height` | | `512` | — | Image height in pixels. |
| `--guidance` | | `1.0` | — | Guidance scale. |
| `--dit` | | — | `OXI_DIT_GGUF`, else `/tmp/parity.gguf` | DiT GGUF path. |
| `--vae` | | — | `OXI_VAE_WEIGHTS`, else `/tmp/bonsai_golden/vae/weights` | VAE weights: a `.safetensors` file or a legacy `.npy` directory. |
| `--te` | | — | `OXI_TE_4BIT`, else `OXI_TE_WEIGHTS`, else `/tmp/bonsai_golden/te/weights` | Text-encoder weights: a 4-bit MLX `model.safetensors` file, or an f32 `.npy` directory. |
| `--tokenizer` | | — | `OXI_TE_TOKENIZER_DIR`, else the TE dir | Tokenizer directory containing `tokenizer.json` (defaults to the TE `.safetensors`' parent). |

---

## Generating an image

### Build with the right GPU feature

Pick the feature that matches your hardware (the GPU backends are opt-in; the
default build is CPU-only Pure Rust):

```bash
# Apple Silicon (Metal — default-on GPU once built with this feature)
cargo build --release --features metal

# NVIDIA (CUDA)
cargo build --release --features native-cuda

# CPU-only (no GPU feature)
cargo build --release
```

The binary lands at `target/release/oxibonsai`. (If you installed via
`cargo install oxibonsai-cli`, the `oxibonsai` binary is already on your PATH.)

### Run

With the four paths set (via `.env` or your shell):

```bash
oxibonsai image \
    --prompt "a tiny bonsai tree in a ceramic pot" \
    --out bonsai.png \
    --seed 42 \
    --steps 4
```

This produces a 512×512 `bonsai.png`. You can pass the paths explicitly instead
of relying on `.env`:

```bash
oxibonsai image \
    --prompt "a tiny bonsai tree in a ceramic pot" --out bonsai.png \
    --dit ./bonsai-dit.gguf \
    --te ./bonsai-te/model.safetensors \
    --vae ./flux2/vae/diffusion_pytorch_model.safetensors \
    --seed 42 --steps 4
```

> **Reproducibility.** `--seed 42` reproduces the official mflux reference
> sample — the noise/ids/schedule are generated by an MLX-exact Threefry RNG port,
> so a given seed maps to the same initial latent as the MLX reference.

---

## Performance

Measured numbers (approximate). The pipeline runs **FP32 in / FP32 accumulate**
throughout, deliberately with **no tensor-core (TF32 / FP16-MAC) paths**, so that
every stage stays at cosine ≥ 0.999 parity with the MLX reference.

| Platform | Backend | Steps | Time / image |
| --- | --- | --- | --- |
| NVIDIA A4000-class | CUDA | 4 | ≈ 31.7 s |
| Apple Silicon (M3-class) | Metal (default-on GPU) | 4 | ≈ 52–62 s |

The CUDA path reached ≈ 31.7 s (≈ **3.2×** faster than the initial ≈ 101 s) after
the GPU-offload work this cycle: a ~6× DiT GEMM speedup, a ~6.3× warp-cooperative
flash-attention, and a ~59× stage-0 context-embedder (ported CPU→GPU), all at
cosine ≥ 0.999. The Metal figure is the full image with the default-on GPU path.

---

## Validation / parity

Every stage is parity-checked against the MLX golden at cosine similarity
**≥ 0.999**. The parity harnesses (run via `cargo run -p oxibonsai-image
--example <name>` with the relevant weights present) verify:

| Harness | What it checks |
| --- | --- |
| `te_parity` | Text-encoder output vs the MLX reference (cos ≥ 0.999). |
| `dit_parity` | DiT forward across all 59 reference taps (each cos ≥ 0.999). |
| `vae_parity` | VAE decode across all 11 reference taps (each cos ≥ 0.999). |
| `vae_safetensors_parity` | The native safetensors loader vs an independent `.npy` reference of the *same* checkpoint, tensor-by-tensor, plus a fixed-latent decode for determinism. |

The native VAE safetensors loader is additionally covered by in-crate unit tests
(`cargo test -p oxibonsai-image --lib vae::safetensors`): the name mapping
(`to_out.0` un-nesting), the conv-weight transpose
(`[O,I,kH,kW] → [O,kH,kW,I]`), and the lossless bf16→f32 decode are each tested
independently. The `vae_safetensors_parity` example then checks that, for every
weight key the decoder requests, the safetensors loader's f32 values match the
independent `.npy` reference dump (it asserts bit-identity), and that decoding a
fixed latent is deterministic (re-decode cosine ≥ 0.999). When a source
`.safetensors` is not available the harness prints `SKIP` and the in-crate unit
tests still apply.

---

## Provenance

To our knowledge, this is the first pure-Rust — C/C++/Fortran-free, zero-FFI —
implementation of the Bonsai-Image FLUX.2-Klein text-to-image pipeline, built
entirely on the COOLJAPAN ecosystem. This echoes the declaration in the project
[`README.md`](../README.md).

---

See also: [`README.md`](../README.md) · [`docs/CLI.md`](CLI.md)
