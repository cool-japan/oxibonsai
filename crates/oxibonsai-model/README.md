# oxibonsai-model

Qwen3 Transformer implementation for 1-bit and ternary Bonsai inference.

Implements the full autoregressive forward pass for the Qwen3 architecture family
(Bonsai-8B/4B/1.7B in Q1_0_g128 and TernaryBonsai-8B/4B/1.7B in TQ2) — token
embedding, Grouped Query Attention with RoPE, SwiGLU MLP, RMSNorm, paged
KV-cache, and Metal/CUDA full-forward integration via `oxibonsai-kernels`.

**Status:** Stable — 1,048 tests passing (`cargo nextest run -p oxibonsai-model`)
**Version:** 0.1.3

Part of the [OxiBonsai](https://github.com/cool-japan/oxibonsai) project.

## Features

### Core Transformer
- `BonsaiModel` — full Qwen3 forward pass: token embedding → N transformer blocks → final RMSNorm → LM head
- `TransformerBlock` — attention sublayer + SwiGLU FFN sublayer with residual connections (`block/`)
- RMSNorm, RoPE (base=1M), Grouped Query Attention (32 Q / 8 KV heads, head_dim=128)
- SwiGLU FFN (gate/up → SiLU × gate → down projection)
- `CausalMask` and sliding-window attention

### Model Variants & Registry
- `ModelVariant::Bonsai{8B, 4B, 1_7B}` — Q1_0_g128 1-bit weights
- `ModelVariant::TernaryBonsai{8B, 4B, 1_7B}` — TQ2 ternary weights
- `ModelSpec`, `CapabilityProfile`, `all_specs()` in `model_variants.rs`
- Architecture auto-detection from GGUF metadata in `model_registry.rs`
- `Qwen3Config` + `ModelConfigBuilder` for custom configs

### Weight Loading
- GGUF loader with tensor-name mapping (`gguf_loader.rs`, `convert/name_map.rs`)
- Q1 loader path via `oxibonsai-kernels` blocks
- `LinearTernary` layer + `load_ternary_blocks` + `load_ternary_embedding` + `OutputWeight::Ternary` (TQ2)
- Safetensors loading support

### KV Cache
- Standard `KvCache` with position-indexed storage
- `PagedKvCache` — vLLM-style block-based cache for high utilization
- `KvCacheFp16` — K/V stored in f16 (halves cache memory)
- `KvCacheQuant` — Q8/Q4 quantized KV cache

### Advanced Attention
- Flash attention / fused kernel (`attention_fused.rs`)
- Flash decoding (`flash_decode.rs`)
- Attention sink (`attention_sink.rs`)
- Cross-attention (`cross_attention.rs`)
- Sparse attention: local window, BigBird, Longformer, dilated (`sparse_attention.rs`)
- ALiBi and YaRN RoPE variants
- RoPE scaling: YaRN, linear, DynamicNTK, LLaMA 3.1, LongRoPE

### Training & Fine-tuning Utilities
- LoRA + LoRA trainer (`lora.rs`, `lora_trainer.rs`)
- Mixture-of-Experts: `MoeRouter`, `MoeExpert`, Mixture-of-Depths
- Optimizers, LR schedulers, losses, gradient / gradient checkpointing
- Pruning, calibration

### Quantization & Export
- Dynamic quantization: DynamicQ8_0, DynamicQ4_0, DynamicQ4_1
- `quantize_int8.rs`, `quantize_ternary.rs` exporters
- `ExportFormat::TernaryG128` in `export.rs`
- Checkpoint save/load (OXCK binary format)

### Model Conversion
- ONNX MatMulNBits (bits=2) ingestion via `oxibonsai convert --onnx` — reads onnx-community Ternary releases and repacks as GGUF TQ2_0_g128
- Qwen3 ONNX tensor role mapping: automatically maps ONNX node names to the Qwen3 weight layout (embedding, QKV projections, gate/up/down FFN, RMSNorm scales)

### Scaling & Inference
- Tensor parallelism, pipeline parallelism, multi-GPU utilities
- Chunked prefill, prefix cache, disk cache
- Weight tying (`TiedEmbedding`) for embedding/LM head sharing
- Model merging: SLERP, TIES, DARE, task vector
- Speculative draft model support
- Compression utilities

## Feature Flags

| Flag | Description |
|------|-------------|
| `wasm` | WASM-safe build |
| `metal` | Metal GPU backend (macOS) — full-forward integration with oxibonsai-kernels |
| `native-cuda` | CUDA GPU backend (NVIDIA) |

## Usage

```toml
[dependencies]
oxibonsai-model = "0.1.3"
```

## License

Apache-2.0 — COOLJAPAN OU
