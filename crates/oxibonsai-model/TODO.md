# oxibonsai-model TODO

> Qwen3 transformer model: layers, blocks, forward pass, KV cache, weight loaders
> ~38,000 lines across `src/`, 1,060+ tests (2026-05-05)

## Status: All Features Complete

Phase 9 ternary support, Phase 10-13 Metal/CUDA full-forward integration, flash
attention, paged KV cache, KV cache quantization, LoRA, MoE, sparse attention,
attention sink, cross-attention, speculative draft model, weight tying, model
merging, and numerical stability tests all implemented and green.

## Done — Core Transformer

- [x] `BonsaiModel` — token embedding, N transformer blocks, final norm, LM head
- [x] `TransformerBlock` — attention sublayer + FFN sublayer with residuals (`block/`)
- [x] RMSNorm layer
- [x] Rotary Position Embeddings (RoPE, base=1M)
- [x] Grouped Query Attention (32 Q heads, 8 KV heads, head_dim=128)
- [x] `CausalMask` — causal attention mask support (`layers/attention.rs`)
- [x] SwiGLU FFN (gate/up projection → SiLU × gate → down projection)
- [x] Sliding window attention (`layers/sliding_window.rs`)
- [x] 1-bit linear layer (Q1_0_g128 weight projection via kernels)
- [x] KV cache with position-indexed storage
- [x] **KV cache FP16 optimization** — K/V stored in f16, halving cache memory (`kv_cache_fp16.rs`)
- [x] **KV cache Q8/Q4 quantization** — `kv_cache_quant.rs`
- [x] **Paged KV cache** — vLLM-style block-based cache (`paged_kv_cache.rs`)
- [x] **Flash attention** — Fused attention kernel to reduce memory bandwidth (`attention_fused.rs`)
- [x] **Flash decoding** — `layers/flash_decode.rs`
- [x] GGUF weight loading with tensor name mapping
- [x] **Bonsai-4B architecture** — config, layers, dims (`model_registry.rs`)
- [x] **Bonsai-1.7B architecture** — smaller variant config (`model_registry.rs`)
- [x] **Architecture auto-detection** — Infer model variant from GGUF metadata (`model_registry.rs`)
- [x] **LayerStats** — Per-layer statistics instrumentation (`block/`)
- [x] **#[instrument] tracing** — Span instrumentation on `forward()` and `TransformerBlock::forward()`
- [x] **Layer-level correctness** — `tests/layer_correctness_tests.rs`: RMSNorm, SwiGLU, RoPE, Attention, TransformerBlock reference comparisons
- [x] **Numerical stability tests** — `tests/numerical_stability_tests.rs`: extreme inputs, overflow/underflow, long sequences, KV cache stress

## Done — Ternary Bonsai (Phase 9)

- [x] `Qwen3Config::ternary_bonsai_{8b,4b,1_7b}()` constructors in `config.rs`
- [x] `ModelVariant::TernaryBonsai{8B,4B,1_7B}` + `from_config_and_sample_tensor_type()` in `model_registry.rs`
- [x] `ternary_bonsai_*_spec()` + capability profiles in `model_variants.rs`
- [x] `LinearTernary` layer + `load_ternary_blocks` + `load_ternary_embedding` + `OutputWeight::Ternary` in `model/weight_loaders.rs` and `layers/linear.rs`
- [x] `ExportFormat::TernaryG128` + `quantize_ternary.rs` exporter in `export.rs`
- [x] Ternary integration tests (`tests/ternary_integration.rs`)

## Done — Phase 10-13 GPU Full-Forward

- [x] Metal full-forward integration via `oxibonsai-kernels` (fused TQ2 ~50 tok/s on 1.7B)
- [x] CUDA full-forward layer parameter handling and weight management
- [x] CUDA inference tests (`tests/cuda_inference_tests.rs`)

## Done — Advanced Attention

- [x] Attention sink (`layers/attention_sink.rs`)
- [x] Cross-attention (`layers/cross_attention.rs`)
- [x] Sparse attention patterns: local window, BigBird, Longformer, dilated (`layers/sparse_attention.rs`)
- [x] ALiBi positional bias (`layers/alibi.rs`)
- [x] YaRN RoPE (`layers/yarn_rope.rs`)
- [x] RoPE scaling variants: YaRN, linear, DynamicNTK, LLaMA 3.1, LongRoPE (`layers/rope_scaling.rs`)
- [x] Mixture-of-Depths (`layers/mixture_of_depths.rs`)

## Done — Training & Fine-tuning

- [x] LoRA (`lora.rs`) + LoRA trainer (`lora_trainer.rs`)
- [x] MoE router + expert (`layers/moe_router.rs`, `layers/moe_expert.rs`)
- [x] Optimizers (`optimizer.rs`)
- [x] LR schedulers (`lr_schedulers.rs`)
- [x] Losses (`losses.rs`)
- [x] Gradient + gradient checkpointing (`gradient.rs`, `gradient_checkpoint.rs`)
- [x] Pruning (`pruning.rs`)
- [x] Calibration (`calibration.rs`)

## Done — Quantization & Export

- [x] Dynamic quantization: DynamicQ8_0, DynamicQ4_0, DynamicQ4_1 (`dynamic_quant.rs`)
- [x] Int8 quantization export (`quantize_int8.rs`)
- [x] Ternary quantization export (`quantize_ternary.rs`)
- [x] Checkpoint save/load — OXCK binary format (`checkpoint.rs`)
- [x] Compression utilities (`compression.rs`)

## Phase 19 — Q2_K / Q3_K / Q4_K / Q8_K Full Stack

- [x] **Q2_K block type** — `BlockQ2K` (84 bytes/256w, 2-bit super-block with 16 K-quant scale nibbles + delta f16) in `oxibonsai-core::quant_k`; `gemv_q2k.rs` scalar GEMV kernel; `LinearQ2K<'a>` in `layers/linear_kquant_full.rs`; `LinearLayer::Q2K` + `OutputWeight::Q2K` variants; Q2_K weight loaders in `weight_loaders.rs`; `forward_cuda.rs` exhaustive match arms
- [x] **Q3_K block type** — `BlockQ3K` (110 bytes/256w, 3-bit with 4-bit signed scale nibbles + hmask high-bit array + d f16) in `oxibonsai-core::quant_k`; `gemv_q3k.rs` scalar GEMV kernel; `LinearQ3K<'a>`; full model integration
- [x] **Q4_K block type** — `BlockQ4K` (144 bytes/256w, 4-bit K-quant with 2-level super-block scale+min; `d`/`dmin` f16) in `oxibonsai-core::quant_k`; `gemv_q4k.rs` scalar GEMV kernel; `LinearQ4K<'a>`; full model integration
- [x] **Q8_K block type** — `BlockQ8K` (292 bytes/256w, 8-bit with f32 scale + precomputed `bsums` i16 block sums) in `oxibonsai-core::quant_k`; `gemv_q8k.rs` scalar GEMV kernel; `LinearQ8K<'a>`; full model integration
- [x] **Lib re-exports** — `pub use layers::linear_kquant_full::{LinearQ2K, LinearQ3K, LinearQ4K, LinearQ8K}` in `lib.rs`

## Phase 18 — Standard GGUF Formats + K-quant Extensions

- [x] **Q4_0 + Q8_0 full stack** — `BlockQ4_0` (18 bytes/32w) + `BlockQ8_0` (34 bytes/32w) in `oxibonsai-core::quant_std`; scalar GEMV kernels; `LinearQ4_0<'a>` + `LinearQ8_0<'a>` in `layers/linear_standard.rs`; `LinearLayer::{Q4_0, Q8_0}` + `OutputWeight::{Q4_0, Q8_0}` variants; Q4_0/Q8_0 weight loaders; 25 block + 8 kernel tests each
- [x] **Q5_K + Q6_K K-quant extensions** — `BlockQ5K` (176 bytes/256w, 5-bit sub-blocks, K-quant scale/min encoding) + `BlockQ6K` (210 bytes/256w, 6-bit symmetric) in `oxibonsai-core::quant_k_ext`; GEMV kernels; `LinearQ5K<'a>` + `LinearQ6K<'a>` in `layers/linear_kquant_ext.rs`; full model integration; 12 tests in `tests/q5k_q6k_model_tests.rs`

## Phase 17 — FP8 KV Cache + SmoothQuant FP8

- [x] **FP8 KV cache** — `Fp8KvLayer`/`Fp8KvFormat`/`Fp8KvCache` in `kv_cache_quant.rs`; per-row abs-max scale encoding; mirrors `QuantizedKvLayer` API; 11 tests in `tests/fp8_kv_cache_tests.rs`; `KvCacheLevel::Fp8` (ordinal 2) in `oxibonsai-runtime`
- [x] **SmoothQuant FP8 calibrator** — `smoothquant.rs`: `SmoothQuantCalibrator` (per-channel `running_max_abs` accumulator across batches), `SmoothQuantError`, `quantize_fp8_e4m3_smooth` / `quantize_fp8_e5m2_smooth`; `BlockFP8E4M3::quantize_with_channel_scales` / `BlockFP8E5M2::quantize_with_channel_scales` in `oxibonsai-core`; 12 tests in `tests/smoothquant_fp8_tests.rs`

## Phase 16 — FP8 Full Stack

- [x] **FP8 export formats (Phase 16C)** — `ExportFormat::FP8E4M3` and `ExportFormat::FP8E5M2`; `BlockFP8E4M3::quantize`-based serialization; GGUF type IDs 43/44; 34B/32w size estimation; 4 tests in `export.rs` (roundtrip E4M3, roundtrip E5M2, size estimate, FP32 exceptions)

## Done — Scaling & Caching

- [x] Tensor parallelism (`tensor_parallel.rs`)
- [x] Pipeline parallelism (`pipeline_parallel.rs`)
- [x] Multi-GPU utilities (`multi_gpu.rs`)
- [x] Chunked prefill (`chunked_prefill.rs`)
- [x] Prefix cache (`prefix_cache.rs`)
- [x] Disk cache (`disk_cache.rs`)
- [x] Weight tying — `TiedEmbedding` for embedding/LM head sharing (`weight_tying.rs`)
- [x] Model merging: SLERP, TIES, DARE, task vector (`model_merge.rs`)
