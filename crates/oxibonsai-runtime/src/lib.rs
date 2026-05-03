//! # OxiBonsai Runtime
//!
//! High-level inference engine, sampling, tokenizer bridge, and
//! OpenAI-compatible HTTP server for OxiBonsai.
//!
//! This crate ties together [`oxibonsai_core`], [`oxibonsai_kernels`],
//! and [`oxibonsai_model`] into a production-ready inference runtime:
//!
//! - **[`InferenceEngine`]** — orchestrates prefill + decode with metrics
//! - **[`Sampler`]** — temperature, top-k, top-p, repetition penalty
//! - **[`SamplingPreset`]** — named parameter sets (Greedy, Balanced, Creative, ...)
//! - **[`SamplerBuilder`] / [`ConfigBuilder`] / [`EngineBuilder`]** — ergonomic setup
//! - **[`server`]** — Axum-based `/v1/chat/completions` server (feature-gated)
//! - **[`InferenceMetrics`]** — Prometheus-compatible counters, gauges, histograms
//! - **[`CircuitBreaker`]** — resilience pattern for cascading-failure protection
//! - **[`HealthReport`]** — structured health checks for ops monitoring
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use oxibonsai_core::config::Qwen3Config;
//! use oxibonsai_runtime::engine::InferenceEngine;
//! use oxibonsai_runtime::presets::SamplingPreset;
//!
//! let config = Qwen3Config::tiny_test();
//! let params = SamplingPreset::Balanced.params();
//! let mut engine = InferenceEngine::new(config, params, 42);
//!
//! let tokens = engine.generate(&[151644, 872], 10)
//!     .expect("generation should succeed");
//! ```

pub mod adaptive_lookahead;
pub mod adaptive_sampling;
#[cfg(feature = "server")]
pub mod admin;
#[cfg(feature = "server")]
pub mod api_extensions;
#[cfg(feature = "server")]
pub mod api_types;
#[cfg(not(target_arch = "wasm32"))]
pub mod async_engine;
pub mod auto_tuner;
pub mod batch_engine;
pub mod beam_search;
pub mod builders;
pub mod circuit_breaker;
#[cfg(feature = "server")]
pub mod completions;
pub mod config;
pub mod constrained_decoding;
pub mod context_manager;
pub mod continuous_batch;
pub mod convenience;
pub mod dedup;
#[cfg(feature = "server")]
pub mod distributed;
pub mod embedding_index;
#[cfg(feature = "server")]
pub mod embeddings;
pub mod engine;
pub mod error;
pub mod health;
pub mod hot_reload;
pub mod json_schema;
pub mod kv_cache_policy;
pub mod memory;
pub mod metrics;
pub mod middleware;
pub mod model_cache;
pub mod multi_model;
pub mod native_tokenizer;
pub mod nbest;
pub mod ngram_cache;
pub mod pipeline;
pub mod prefix_cache_engine;
pub mod presets;
pub mod profiler;
pub mod quality_metrics;
#[cfg(feature = "rag")]
pub mod rag_server;
pub mod rate_limiter;
pub mod recovery;
pub mod request_id;
pub mod request_metrics;
pub mod request_queue;
pub mod sampling;
pub mod sampling_advanced;
pub mod semantic_cache;
#[cfg(feature = "server")]
pub mod server;
pub mod speculative;
pub mod stream_metrics;
pub mod streaming;
pub mod token_budget;
pub mod token_healing;
pub mod tokenizer_bridge;
pub mod tracing_setup;
pub mod wasm_api;
#[cfg(feature = "server")]
pub mod web_ui;

pub use adaptive_lookahead::{AdaptiveLookahead, AdaptiveLookaheadConfig, AdaptiveLookaheadError};
pub use adaptive_sampling::{
    AdaptiveSamplerChain, AdaptiveStrategy, EntropyCooling, GenerationState, RepetitionAdaptation,
    ScheduledDecay,
};
pub use auto_tuner::{
    AutoTuner, CpuArch, CpuFeatures, KernelBenchmark, KvCacheType, MemoryBudget, SimdTier,
    TuningRecommendation,
};
pub use builders::{ConfigBuilder, EngineBuilder, SamplerBuilder};
pub use circuit_breaker::{CircuitBreaker, CircuitBreakerConfig, CircuitState};
pub use config::OxiBonsaiConfig;
pub use constrained_decoding::{
    ConstrainedSampler, ConstrainedSamplerBuilder, ConstraintError, JsonConstraint, JsonParseState,
    NoConstraint, RegexConstraint, TokenConstraint,
};
pub use convenience::{GenerationResult, MemoryEstimate, ModelFileInfo, TokenStats};
pub use dedup::{DedupCache, DedupStats, RequestKey};
#[cfg(feature = "server")]
pub use distributed::{
    ConsistentHashRing, CoordinatorConfig, DistributedCoordinator, NodeInfo, NodeRegistry,
};
pub use engine::InferenceEngine;
pub use error::{RuntimeError, RuntimeResult};
pub use health::{HealthReport, HealthStatus};
pub use hot_reload::{HotReloadCoordinator, ModelVersion, ReloadLog};
pub use json_schema::{
    parse_schema, schema_example, schema_template, validate_against_schema, SchemaError,
    SchemaState, SchemaType,
};
pub use kv_cache_policy::{KvCacheLevel, KvCachePolicy, KvCachePolicyConfig, KvCachePolicyError};
pub use memory::{get_rss_bytes, MemoryProfiler, MemorySnapshot};
pub use metrics::InferenceMetrics;
pub use multi_model::{
    AdapterRef, AdapterStack, EndpointStatus, ModelEndpoint, ModelId, ModelListEntry,
    ModelRegistry, ModelRouter, RoutingError,
};
pub use native_tokenizer::{NativeTokenizerBridge, NativeTokenizerError};
pub use nbest::{Hypothesis, NBestDecoder, NBestList};
pub use presets::SamplingPreset;
pub use profiler::{flop_counter, AggregateStats, ProfileEvent, ProfileTrace, Profiler};
pub use quality_metrics::{
    extract_ngrams, perplexity_from_logprobs, repetition_penalty_rate, self_bleu, token_entropy,
    BatchQualityAnalyzer, BleuScore, DiversityMetrics, GenerationQualityReport, RepetitionMetrics,
};
pub use recovery::{ErrorClass, RecoveryStrategy};
pub use request_id::RequestId;
pub use request_metrics::{
    AggregateRateSnapshot, RequestRateAggregator, RequestRateSnapshot, RequestRateTracker,
};
pub use sampling::Sampler;
pub use sampling_advanced::{
    EtaSampler, LcgRng, MinPSampler, MirostatV1Sampler, MirostatV2Sampler, SamplerChain,
    SamplerStep, TypicalSampler,
};
pub use stream_metrics::{RequestStreamMetrics, StreamMetricsSnapshot, StreamingMetricsAggregator};
pub use token_budget::{
    BudgetConfig, BudgetError, BudgetPolicy, GlobalTokenBudget, RequestBudget, TokenCostEstimate,
};
pub use tokenizer_bridge::TokenizerBridge;
pub use tracing_setup::{init_tracing, TracingConfig};
#[cfg(feature = "server")]
pub use web_ui::create_ui_router;
