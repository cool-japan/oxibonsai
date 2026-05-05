//! # oxibonsai-model
//!
//! Qwen3 Transformer implementation for 1-bit Bonsai inference.
//!
//! This crate implements the full autoregressive forward pass for the
//! Qwen3 architecture family (8B, 4B, 1.7B) using 1-bit quantised
//! weights. The forward pass pipeline is:
//!
//! 1. **Token embedding** — FP32 lookup from a `[vocab_size x hidden_size]` table
//! 2. **N Transformer blocks**, each containing:
//!    - Pre-attention **RMSNorm**
//!    - **Grouped Query Attention** (GQA) with rotary position embeddings
//!    - Pre-FFN **RMSNorm**
//!    - **SwiGLU MLP** (gate + up + down projections)
//! 3. **Final RMSNorm**
//! 4. **LM head** projection to vocabulary logits
//!
//! All linear projections in the Transformer blocks use Q1\_0\_g128 1-bit
//! weights dispatched through [`oxibonsai_kernels::OneBitKernel`].
//!
//! ## Model Registry
//!
//! [`ModelVariant`] auto-detects the architecture from configuration
//! dimensions and provides parameter counts and expected file sizes.

pub mod block;
pub mod calibration;
pub mod checkpoint;
pub mod chunked_prefill;
pub mod compression;
pub mod convert;
pub mod disk_cache;
pub mod dynamic_quant;
pub mod error;
pub mod export;
pub mod gguf_loader;
pub mod gradient;
pub mod gradient_checkpoint;
pub mod kv_cache;
pub mod kv_cache_fp16;
pub mod kv_cache_quant;
pub mod layers;
pub mod lora;
pub mod lora_trainer;
pub mod losses;
pub mod lr_schedulers;
pub mod model;
pub mod model_config_builder;
pub mod model_merge;
pub mod model_registry;
pub mod model_variants;
pub mod multi_gpu;
pub mod optimizer;
pub mod paged_kv_cache;
pub mod pipeline_parallel;
pub mod prefix_cache;
pub mod pruning;
pub mod quantize;
pub mod quantize_int8;
pub mod quantize_ternary;
pub mod smoothquant;
pub mod tensor_parallel;
pub mod weight_tying;

pub use calibration::{
    simulate_calibration, validate_calibration, CalibMethod, CalibSummary, CalibValidation,
    CalibrationDb, LayerCalibStats,
};
pub use checkpoint::{Checkpoint, CheckpointError, CheckpointMetadata, CheckpointTensor};
pub use chunked_prefill::{
    create_prefill_chunks, peak_memory_estimate, ChunkedPrefillConfig, PrefillAction, PrefillChunk,
    PrefillMemoryEstimate, PrefillPriority, PrefillScheduler,
};
pub use compression::{
    compress_model, estimate_compressed_size, CompressionConfig, CompressionError,
    CompressionResult, CompressionStage, StageStats,
};
pub use disk_cache::{
    CacheEntry, CacheFileInfo, CacheManager, DiskCache, DiskCacheError, CACHE_MAGIC, CACHE_VERSION,
};
pub use dynamic_quant::{
    compute_scale, compute_smooth_factors, dynamic_quantize_int4, dynamic_quantize_int8,
    dynamic_quantize_int8_per_row, quantization_mae, smooth_activations, smooth_weights,
    w8a8_matvec, CalibStats, DynQuantError, DynQuantFormat, DynQuantTensor, DynamicScaleMode,
    SmoothQuantConfig,
};
pub use error::{ModelError, ModelResult};
pub use smoothquant::{
    quantize_fp8_e4m3_smooth, quantize_fp8_e5m2_smooth, SmoothQuantCalibrator, SmoothQuantError,
};
pub use gguf_loader::{
    estimate_memory_bytes, fits_in_budget, load_tensor_metadata, validate_gguf_file, LoadConfig,
    LoadError, LoadStats, TensorChunkIter, TensorEntry,
};
pub use gradient_checkpoint::{
    Checkpoint as GradientCheckpoint, CheckpointBudget, CheckpointError as GradientCheckpointError,
    CheckpointSegment, CheckpointStrategy, CheckpointedActivation, CheckpointedNetwork,
    CheckpointedPipeline, LinearSegment, Recomputable,
};
pub use kv_cache::KvCache;
pub use kv_cache_fp16::KvCacheFp16;
pub use kv_cache_quant::{
    dequantize_row_i8, quant_error_mae, quantize_row_i8, Fp8KvCache, Fp8KvFormat, Fp8KvLayer,
    QuantKvError, QuantizedKvCache, QuantizedKvLayer,
};
pub use layers::attention_sink::{
    AttentionSinkCache, AttentionSinkConfig, AttentionSinkLayer, SinkError, SinkSlot,
};
pub use layers::cross_attention::{
    causal_cross_attention, compute_attention_weights, cross_attention_forward,
    single_head_cross_attention, CrossAttentionConfig, CrossAttnError,
};
pub use layers::flash_decode::{
    flash_decode_multi_head, flash_decode_single_head, flash_vs_naive_error, FlashDecodeConfig,
    FlashDecodeError,
};
pub use layers::mixture_of_depths::{
    mixture_of_depths_forward, ModConfig, ModError, ModRouter, ModStats,
};
pub use layers::rope_scaling::{
    apply_rope_with_freqs, compute_rope_frequencies, dynamic_ntk_base, llama31_frequencies,
    FreqStats, RopeScalingError, RopeScalingStrategy,
};
pub use layers::sparse_attention::{
    memory_reduction, sparse_attention_forward, sparse_vs_dense_error, SparseAttentionMask,
    SparseAttnError, SparsePattern,
};
pub use layers::yarn_rope::{
    apply_rope, apply_yarn_rope, LongRopeConfig, YarnConfig, YarnError, YarnFreqTable,
};
pub use losses::{
    contrastive_loss, cross_entropy, cross_entropy_grad, cross_entropy_single, distillation_loss,
    focal_loss, huber_loss, kl_divergence, label_smoothed_cross_entropy, log_softmax, mse,
    ntp_loss, softmax, LossError,
};
pub use lr_schedulers::{
    CyclicLr, LinearWarmupCosineDecay, OneCycleLr, PlateauMode, PolynomialDecay, ReduceOnPlateau,
};
pub use model::BonsaiModel;
pub use model_merge::{
    dare_merge, linear_merge, merge_models, merge_models_with_stats, merge_tensors, slerp,
    task_vector_merge, ties_merge, MergeConfig, MergeError, MergeMethod, MergeStats, WeightTensor,
};
pub use model_registry::ModelVariant;
pub use multi_gpu::{
    merge_column_shards, partition_weights_column, partition_weights_row, CollectiveResult,
    DeviceId, DeviceInfo, DeviceMesh, NcclCollectives,
};
pub use paged_kv_cache::{
    BlockPool, BlockTable, KvPage, PagedKvCache, PagedKvError, DEFAULT_BLOCK_SIZE,
};
pub use prefix_cache::{
    CacheBlock, CacheSession, PrefixAwarePrefill, PrefixCache, PrefixCacheStats,
};
pub use pruning::{
    compute_importance, model_sparsity_report, prune_model, prune_tensor, prune_tensor_inplace,
    ImportanceMetric, ImportanceScores, ModelSparsitySummary, PruningConfig, PruningError,
    PruningGranularity, ScoreStats, SparsityReport,
};
pub use weight_tying::{TiedEmbedding, TyingError};

pub use convert::onnx::{convert_onnx_to_gguf, DequantError as OnnxDequantError, OnnxImportError};
pub use convert::ConvertStats;
