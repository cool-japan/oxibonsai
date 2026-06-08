# oxibonsai-image

Pure-Rust text-to-image pipeline: FLUX.2-Klein DiT (TQ2_0_g128 ternary) + AutoencoderKLFlux2 VAE + Qwen3-4B 4-bit text encoder + PNG output, all parity-validated against the MLX reference at cosine ≥ 0.999.

**Version:** 0.2.3

Part of the [OxiBonsai](https://github.com/cool-japan/oxibonsai) project.

---

## What it does

`oxibonsai-image` implements the complete Bonsai-Image text-to-image pipeline in
Pure Rust. Given a text prompt, it runs:

```text
prompt ──▶ Text Encoder ──▶ DiT ──▶ VAE decoder ──▶ PNG
           (Qwen3-4B,        (ternary FLUX.2    (AutoencoderKLFlux2)
            4-bit MLX)        transformer,
                              TQ2_0_g128)
```

| Stage | Model | On-disk format |
|-------|-------|----------------|
| Text Encoder | Qwen3-4B, 4-bit | MLX 4-bit `.safetensors` (≈2.1 GB) |
| DiT | FLUX.2-Klein ternary transformer | GGUF, `TQ2_0_g128` |
| VAE decoder | AutoencoderKLFlux2 | FLUX.2 `.safetensors` |
| Tokenizer | Qwen3 BPE | `tokenizer.json` |
| PNG encode | oxiarc-deflate | PNG |

Every stage is parity-validated against the MLX golden reference:

| Harness | Gate |
|---------|------|
| `te_parity` | Text-encoder output cosine ≥ 0.999 |
| `dit_parity` | DiT forward across all 59 reference taps, each cosine ≥ 0.999 |
| `vae_parity` | VAE decode across all 11 reference taps, each cosine ≥ 0.999 |
| `vae_safetensors_parity` | Native safetensors loader vs `.npy` reference, bit-identical weights |

---

## Feature Flags

| Flag | Backend | Default | Notes |
|------|---------|---------|-------|
| `metal` | Apple Silicon GPU (Metal) | on (when built with `--features metal`) | Enables TQ2_0_g128 DiT matmuls + VAE + flash-attention on macOS via `oxibonsai-kernels`. GPU is **default-on** at runtime once the feature is compiled in. |
| `native-cuda` | NVIDIA GPU (CUDA) | off | Linux/Windows only. Enables cudarc-backed TQ2 GEMM, warp-cooperative flash-attention, and stage-0 GPU context embedding. |
| *(none)* | CPU only | — | Pure-Rust Rayon+NEON fallback on every stage. Always available regardless of feature selection. |

`metal` and `native-cuda` are mutually exclusive by target platform — the Metal
path is `cfg(target_os = "macos")` and the CUDA path is
`cfg(any(target_os = "linux", target_os = "windows"))`.

---

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
oxibonsai-image = { version = "0.2.3", features = ["metal"] }  # or "native-cuda"
```

Build with the matching GPU feature:

```bash
# Apple Silicon
cargo build --release --features metal

# NVIDIA
cargo build --release --features native-cuda

# CPU-only
cargo build --release
```

### Library Usage

```rust,no_run
use std::path::PathBuf;
use oxibonsai_image::{
    pipeline::{text_to_image, TeSource, TextToImageCfg},
};

let cfg = TextToImageCfg {
    prompt: "a tiny bonsai tree in a ceramic pot".to_string(),
    seed: 42,
    steps: 4,
    width: 512,
    height: 512,
    guidance: 1.0,
    dit_gguf: PathBuf::from("./bonsai-dit.gguf"),
    vae_weights_dir: PathBuf::from("./bonsai-vae/vae/diffusion_pytorch_model.safetensors"),
    te_source: TeSource::Mlx4bit(PathBuf::from("./bonsai-te/text_encoder-mlx-4bit/model.safetensors")),
    tokenizer_dir: PathBuf::from("./bonsai-te/text_encoder-mlx-4bit"),
    golden_override: None,
};

let out = text_to_image(cfg).expect("pipeline failed");
std::fs::write("bonsai.png", &out.png).expect("write png");
println!("Generated {}×{} PNG", out.width, out.height);
```

`text_to_image` returns a [`TextToImageOut`] containing the PNG byte stream,
image dimensions, and per-stage cosine similarities vs the golden (when
[`GoldenOverride`] is set). All errors are surfaced as [`PipelineError`] — no
`unwrap` / `panic`.

---

## Environment Variables

Asset paths can be set via a `.env` file or shell environment variables (CLI
flags take precedence). Create a `.env` in your working directory:

```dotenv
# DiT GGUF (produced by mlx_image_convert from the ternary safetensors)
OXI_DIT_GGUF=./bonsai-dit.gguf

# Text encoder: 4-bit MLX model.safetensors (≈2.1 GB)
OXI_TE_4BIT=./bonsai-te/text_encoder-mlx-4bit/model.safetensors

# Tokenizer directory containing tokenizer.json
# (defaults to the TE .safetensors parent when omitted)
OXI_TE_TOKENIZER_DIR=./bonsai-te/text_encoder-mlx-4bit

# VAE weights: .safetensors file or legacy .npy directory
OXI_VAE_WEIGHTS=./bonsai-vae/vae/diffusion_pytorch_model.safetensors
```

**GPU stage toggles** (default on; set to `"0"` to opt out):

| Variable | Stage |
|----------|-------|
| `OXI_DIT_ATTN_GPU` | DiT joint flash-attention (Metal / CUDA) |
| `OXI_VAE_GPU` | VAE decode (Metal / CUDA) |
| `OXI_TE_GPU` | Text-encoder GEMM (Metal; dormant — set `OXI_TE_GPU=1`) |

See [`docs/IMAGEN.md`](../../docs/IMAGEN.md) for the full environment-variable
and flag reference, including the complete asset-acquisition walkthrough.

---

## Asset Acquisition

You need three model assets plus a tokenizer. Downloads use the HuggingFace CLI
(`pip install huggingface_hub`); all conversion and inference are Pure Rust.

```bash
# 1. DiT: download ternary MLX checkpoint and convert to GGUF
#    (hf download keeps the repo subfolder, so files land under ./bonsai-dit/transformer-packed-mflux/)
hf download prism-ml/bonsai-image-ternary-4B-mlx-2bit \
    transformer-packed-mflux/diffusion_pytorch_model.safetensors --local-dir ./bonsai-dit
cargo run -p oxibonsai-model --example mlx_image_convert --release -- \
    ./bonsai-dit/transformer-packed-mflux/diffusion_pytorch_model.safetensors ./bonsai-dit.gguf tq2_0_g128

# 2. Text encoder + tokenizer (same repo; no conversion — native 4-bit Rust loader)
hf download prism-ml/bonsai-image-ternary-4B-mlx-2bit \
    text_encoder-mlx-4bit/model.safetensors text_encoder-mlx-4bit/tokenizer.json \
    --local-dir ./bonsai-te

# 3. VAE (no conversion — native safetensors Rust loader).
#    Option A (simplest, non-gated): the VAE bundled in the same PrismML repo
hf download prism-ml/bonsai-image-ternary-4B-mlx-2bit \
    vae/diffusion_pytorch_model.safetensors --local-dir ./bonsai-vae
#    Option B (canonical, gated — needs huggingface-cli login + license):
# hf download black-forest-labs/FLUX.2-dev \
#     vae/diffusion_pytorch_model.safetensors --local-dir ./flux2
```

For the full step-by-step walkthrough see [`docs/IMAGEN.md`](../../docs/IMAGEN.md).

---

## Performance

Measured at 512×512, 4 Euler steps, FP32 accumulate throughout (no TF32/FP16-MAC
shortcuts, preserving cosine ≥ 0.999 parity):

| Platform | Backend | Time / image |
|----------|---------|-------------|
| Apple Silicon (M3-class) | Metal (default-on GPU) | ≈ 52–62 s |
| NVIDIA A4000-class | CUDA | ≈ 31.7 s |
| Any | CPU only (Rayon + NEON) | ≈ 10–15 min |

GPU acceleration is composed of three independently validated kernels: v10 TQ2
ternary GEMM (≈3.8× over v9), joint flash-attention (≈5.47× over CPU, simdgroup
f32 MACs, flash-v2 online softmax), and implicit-GEMM im2col-free conv for the
VAE (≈3.2× over CPU). All three default to on; the CPU fallback is always
available.

---

## Pure Rust Declaration

`oxibonsai-image` is part of the COOLJAPAN ecosystem and is C/C++/Fortran-free
— zero FFI in the default build; every dependency is Pure Rust.

---

## License

Apache-2.0 — [COOLJAPAN OU](https://github.com/cool-japan/oxibonsai)
