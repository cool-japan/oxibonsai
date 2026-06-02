# oxibonsai-image TODO

> v0.1.5 — 2026-06-02
> **STABLE** — imagen pipeline complete, GPU-accelerated, parity-validated.

## Completed

- [x] DiT forward (FLUX.2-Klein TQ2_0_g128, 5 double + 20 single stream blocks, 4-axis RoPE)
- [x] VAE decoder (AutoencoderKLFlux2, GroupNorm/32, conv2d, SiLU, 4-stage upsample)
- [x] Text Encoder (Qwen3-4B 4-bit MLX safetensors loader, hidden-state extraction, 7680-dim context)
- [x] PNG output (oxiarc-deflate, Pure Rust, DEFLATE L9)
- [x] MLX-exact Threefry RNG (seed reproduces mflux reference output byte-exactly)
- [x] Native VAE safetensors loader (`.safetensors` direct-load, no Python export required)
- [x] `oxibonsai image` CLI subcommand (`--prompt`, `--seed`, `--out`, `--steps`, `--size`)
- [x] Metal GPU acceleration (default-on):
  - DiT flash-attention kernel: 5.47× over CPU rayon+NEON (59ms vs 323ms)
  - DiT ternary GEMM v10 (f16-D staging): 1.89× DiT sampler speedup (34.2s vs 64.7s)
  - VAE implicit-GEMM conv + on-GPU GroupNorm: 3.2× over CPU VAE (6.9s vs 22.5s)
  - Full pipeline: ~52–62s end-to-end (Metal, 512×512, steps=4)
- [x] CUDA GPU acceleration (oxibonsai-kernels NVRTC backend):
  - 3.2× overall vs CPU; steps=4 ≈ 31.7s end-to-end
- [x] Parity validation: `te_parity` (oracle cos≥0.999999), `dit_parity` (59 taps cos≥0.999), `vae_parity` (11 taps cos≥0.999)
- [x] docs/IMAGEN.md + docs/CLI.md

## Deferred

- [ ] VAE tiling for images larger than 512px (activation memory reduction for high-res)
- [ ] TE GPU weight cache for multi-image throughput (amortize 4-bit MLX load across prompts)
- [ ] VAE full GPU residency on CUDA discrete GPU (~4s additional win per image)
- [ ] Binary (1-bit) DiT variant support (not yet available from PrismML)
