# oxibonsai-model TODO

> Qwen3 transformer model: layers, blocks, forward pass, KV cache, weight loaders
> ~38,000 lines across `src/`, 1,048 tests (2026-05-03)

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

## Done — Scaling & Caching

- [x] Tensor parallelism (`tensor_parallel.rs`)
- [x] Pipeline parallelism (`pipeline_parallel.rs`)
- [x] Multi-GPU utilities (`multi_gpu.rs`)
- [x] Chunked prefill (`chunked_prefill.rs`)
- [x] Prefix cache (`prefix_cache.rs`)
- [x] Disk cache (`disk_cache.rs`)
- [x] Weight tying — `TiedEmbedding` for embedding/LM head sharing (`weight_tying.rs`)
- [x] Model merging: SLERP, TIES, DARE, task vector (`model_merge.rs`)
