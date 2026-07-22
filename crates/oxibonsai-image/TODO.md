# oxibonsai-image TODO

> v0.2.3 — 2026-07-21
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

- [x] VAE tiling for images larger than 512px (activation memory reduction for high-res) (done 2026-06-17)
  - **Goal:** Tiled decode path with peak activation memory O(tile_size) instead of O(px²). The untiled 512² peak is ~3.6 GB im2col; 1024² would be ~14.5 GB (OOM). Result is numerically identical to untiled: cos ≥ 0.999 vs the untiled `vae_decoded` golden.
  - **Design:** Two boundaries — `TileBoundary::AfterConvNormOut` (default, seam-free by construction: run everything through `conv_norm_out` whole for correct global GroupNorm stats, then tile only `silu→conv_out` with halo=1px, exact-overlap, no blend) and `TileBoundary::AfterMid` (two-pass GroupNorm: Pass 1 computes global per-group stats, Pass 2 applies them per-tile; keeps `post_quant_conv→conv_in→mid_block` whole since mid has global attention; caps the real 3.6 GB up2 peak to ~1 GB at tile 256²). New `GroupStats`, `compute_stats`, `apply_precomputed` on `norm.rs` (additive, `forward_inplace` delegates). Window extraction zero-fills global-edge out-of-bounds indices (matches `build_im2col` zero-pad), pulls real neighbor data at interior seams. New `TileConfig { tile_px: 256, boundary }` + `decode_packed_latents_tiled` (non-breaking). Auto-tile in `pipeline.rs`/`session.rs` only when `16*ph > 512`. Per-op Metal/CUDA dispatch unchanged.
  - **Files:** `src/vae/norm.rs` (~70), `src/vae/decoder.rs` (~260) or new `src/vae/tiling.rs` (~300), `src/vae/mod.rs` (~6), `src/pipeline.rs` (~6), `src/session.rs` (~6), new `examples/vae_tiled_parity.rs` (~180). ~530 LoC.
  - **Tests:** `examples/vae_tiled_parity.rs`: decode 32×32 latent golden with `tile_px=128` (forces 16 tiles), both boundaries, assert cos ≥ 0.999 + relL2 ≤ 2e-2 vs untiled golden AND in-process cross-check cos ≥ 0.99999 tiled-vs-untiled. Unit `#[test]`s: `groupnorm_two_pass_equals_one_pass`, `tiled_conv_out_equals_untiled` (halo=1, edge-zero-pad vs seam-real-data), `tile_grid_covers_canvas` (every output pixel written exactly once). Run with `OXI_VAE_GPU=0` to prove CPU path.
  - **Risk:** GroupNorm two-pass is the crux — mitigated by unit-testing `compute_stats`/`apply_precomputed` independently before wiring `AfterMid`.
  - **Follow-up (2026-07-21):** a production-release audit found `TileBoundary::AfterMid` was actually a no-op — it fell through to the same whole-up-block path as `AfterConvNormOut`, so it never bounded the up-block im2col memory it exists for. Fixed for real in `vae/decoder.rs` (`decode_packed_latents_tiled` now dispatches `AfterMid` to `decode_before_conv_out_tiled`/`UpBlock::forward_tiled`, budgeted to `tile_px²` pixels); covered by regression tests.
- [x] TE GPU weight cache for multi-image throughput (amortize 4-bit MLX load across prompts) (done 2026-06-17)
  - **Goal:** Make the Metal GPU TE weight cache residency-aware: evict-when-non-resident (closes a latent stale-handle hazard where recycled `as_ptr()` keys return stale GPU buffers → corrupted conditioning on `OXI_TE_GPU=1`), persist-when-resident (explicit cross-prompt amortization). Add upload counter for testability.
  - **Design:** Add `pub fn is_resident(&self) -> bool` to `TeWeights`. Add symmetric Metal `MetalGraph::evict_f32_weight(&self, key) -> Result<(), MetalGraphError>` (mirrors CUDA's existing `evict_f32_weight`). Add `f32_upload_count: AtomicUsize` to `MetalGraph`, increment on insert branch, expose `pub fn weight_upload_count() -> usize`. Thread `resident: bool` from `TextEncoder` (holds `&TeWeights`) through `matmul → matmul_inner → te_matmul_gpu`; in `te_matmul_gpu` when not-resident evict after GEMM, when resident leave cached. Pure helper `fn should_persist(resident: bool) -> bool` for unit testability. Stays out of `pipeline.rs`/`session.rs`.
  - **Files:** `src/te/weights.rs` (~6), `src/te/forward.rs` (~12), `src/te/gpu.rs` (~15), `oxibonsai-kernels` `metal_graph/graph.rs` + `metal_full_layer/functions_2.rs` (~25). ~58 LoC.
  - **Tests:** `test_should_persist_logic` (pure-logic, no GPU). GPU reuse test (`#[ignore]`, run `--test-threads=1`): resident `TeWeights`, encode same prompt twice, assert `weight_upload_count()` unchanged on second pass.
  - **Risk:** Central crate (`oxibonsai-kernels`) touched; opt-in path only (`OXI_TE_GPU=1`, default-off). Evict-when-non-resident is strictly safer than today.
  - **Follow-up (2026-07-21):** extended real bounded-LRU residency to the CUDA backend too (`te/cuda_gpu.rs`), which previously only had per-call `evict_f32_weight` (no persistent cache). New `OXI_TE_GPU_RESIDENT_BUDGET_MB` (default `0` = original evict-after-every-GEMM behavior) lets a resident session keep up to N MB of TE device weights cached across prompts, LRU-evicted beyond budget — VRAM-bounded since the ~16 GB f32 encoder can't stay fully resident on a discrete GPU the way it does under Metal's unified memory.
- [ ] VAE full GPU residency on CUDA discrete GPU (~4s additional win per image)
- [ ] Binary (1-bit) DiT variant support (not yet available from PrismML)
