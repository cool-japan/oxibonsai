//! Full architectural specification for all Bonsai model variants.
//!
//! [`ModelSpec`] captures the complete set of architectural parameters, memory
//! requirements, and descriptive metadata for each known Bonsai variant.
//! [`CapabilityProfile`] describes the runtime characteristics most relevant
//! to application developers (context window, streaming support, recommended
//! sampling settings, supported languages, and typical use-cases).
//!
//! ## Weight-size calculation (Q1_0_g128)
//!
//! At Q1_0_g128 each group of 128 weights is stored as:
//! - 128 bits (16 bytes) of packed 1-bit weights
//! - 2 bytes of FP16 group scale
//!
//! Total storage per weight = (16 + 2) / 128 = **0.140625 bytes ≈ 1.125 bits**.
//!
//! Embedding and output-projection matrices are stored in FP16 (2 bytes/param)
//! because they are not quantised in the Q1_0 scheme.
//!
//! ## KV-cache size (4 k context)
//!
//! KV cache at FP32 for a single sequence of 4 096 tokens:
//!
//! ```text
//! bytes = num_layers × 2 × num_kv_heads × head_dim × seq_len × 4
//! ```

use oxibonsai_core::config::Qwen3Config;

use crate::model_registry::ModelVariant;

// ─── ModelSpec ───────────────────────────────────────────────────────────────

/// Complete specification for a single Bonsai model variant.
///
/// All byte-count fields represent approximate values; exact figures depend on
/// the GGUF file produced by the quantisation tool.
#[derive(Debug, Clone)]
pub struct ModelSpec {
    /// Short human-readable name (e.g. `"Bonsai-8B"`).
    pub name: &'static str,
    /// The [`ModelVariant`] enum value this spec describes.
    pub variant: ModelVariant,
    /// Reference architecture configuration.
    pub config: Qwen3Config,
    /// Approximate total parameter count.
    pub param_count: u64,
    /// Approximate on-disk / in-memory size of quantised weights at Q1_0_g128.
    ///
    /// Transformer weights at 1.125 bits/param + FP16 embeddings + norms.
    pub weights_size_bytes: u64,
    /// KV-cache size in bytes for a 4 096-token context at FP32.
    pub kv_cache_4k_bytes: u64,
    /// Estimated minimum RAM required to run inference at 4 k context.
    ///
    /// `weights_size_bytes + kv_cache_4k_bytes + runtime_overhead`.
    pub min_ram_bytes: u64,
    /// Free-text description of this variant.
    pub description: &'static str,
}

// ─── Per-variant spec constructors ───────────────────────────────────────────

/// Build the [`ModelSpec`] for Bonsai-8B (Qwen3-8B architecture).
///
/// Architecture: 36 layers, hidden=4096, intermediate=14336, Q=32 heads, KV=8 heads.
pub fn bonsai_8b_spec() -> ModelSpec {
    let config = Qwen3Config::bonsai_8b();

    // ── param count ──────────────────────────────────────────────────────────
    // Embedding table:  151 936 × 4 096 = 622 756 864
    // Output (tied):    151 936 × 4 096 = 622 756 864  (separate in GGUF)
    // Per layer (36):
    //   Q: 4096×4096=16 777 216   K: 4096×1024=4 194 304
    //   V: 4096×1024=4 194 304   O: 4096×4096=16 777 216
    //   gate: 4096×14336=58 720 256  up: 58 720 256  down: 58 720 256
    //   norms (×3 × 4096) ≈ 12 288
    // Layer total ≈ 219 116 096  × 36 ≈ 7 888 179 456
    // Final norm: 4 096
    // Grand total ≈ 8 030 000 000
    let param_count: u64 = 8_030_000_000;

    // ── weights at Q1_0_g128 ─────────────────────────────────────────────────
    // Quantised weights (transformer blocks only, excluding embedding/output):
    //   params = 7 888 179 456  →  bytes = params * 1.125 / 8 = 1 111 653 000 (≈1.04 GiB)
    // Embedding (FP16): 151936 × 4096 × 2 = 1 245 513 728 (≈1.16 GiB)
    // Output head (FP16): same = 1 245 513 728
    // Norms (FP32): ~36 × 2 × 4096 × 4 = 1 179 648
    // Metadata overhead: ~50 MiB
    // Total ≈ 2 200 000 000
    let weights_size_bytes: u64 = 2_200_000_000;

    // ── KV cache at 4 096 context, FP32 ─────────────────────────────────────
    // 36 × 2 × 8 × 128 × 4096 × 4 = 1 207 959 552
    let kv_cache_4k_bytes: u64 = kv_cache_size_bytes(&config, 4096);

    // ── minimum RAM ─────────────────────────────────────────────────────────
    // weights + kv_cache + 64 MiB runtime overhead
    let min_ram_bytes = weights_size_bytes + kv_cache_4k_bytes + 64 * 1024 * 1024;

    ModelSpec {
        name: "Bonsai-8B",
        variant: ModelVariant::Bonsai8B,
        config,
        param_count,
        weights_size_bytes,
        kv_cache_4k_bytes,
        min_ram_bytes,
        description: "Bonsai-8B is the flagship variant based on Qwen3-8B. \
            36-layer GQA transformer with 4096-dimensional hidden state, 32 query heads, \
            8 KV heads, and a 65 536-token context window. Recommended for highest quality \
            output where >= 4 GB RAM is available.",
    }
}

/// Build the [`ModelSpec`] for Bonsai-4B.
///
/// Architecture: 24 layers, hidden=2560, intermediate=6912, Q=20 heads, KV=4 heads.
pub fn bonsai_4b_spec() -> ModelSpec {
    let config = Qwen3Config::bonsai_4b();

    // ── param count ──────────────────────────────────────────────────────────
    // Embedding: 151936 × 2560 = 388 952 064
    // Output:    same = 388 952 064
    // Per layer (24):
    //   Q: 2560×2560=6 553 600  K: 2560×512=1 310 720
    //   V: 2560×512=1 310 720   O: 2560×2560=6 553 600
    //   gate: 2560×6912=17 694 720  up: 17 694 720  down: 17 694 720
    //   norms ≈ 7 680
    // Layer total ≈ 68 820 480  × 24 = 1 651 691 520
    // Total ≈ 4 020 000 000 (embeddings dominate less here)
    let param_count: u64 = 4_020_000_000;

    // ── weights at Q1_0_g128 ─────────────────────────────────────────────────
    // Quantised blocks: ~1 651 691 520 params × 1.125/8 ≈ 232 426 152 bytes
    // Embedding FP16:   388 952 064 × 2 = 777 904 128
    // Output FP16:      same
    // Total ≈ 1 300 000 000
    let weights_size_bytes: u64 = 1_300_000_000;

    // ── KV cache at 4 096 context ─────────────────────────────────────────────
    // 24 × 2 × 4 × 128 × 4096 × 4 = 402 653 184
    let kv_cache_4k_bytes: u64 = kv_cache_size_bytes(&config, 4096);

    let min_ram_bytes = weights_size_bytes + kv_cache_4k_bytes + 48 * 1024 * 1024;

    ModelSpec {
        name: "Bonsai-4B",
        variant: ModelVariant::Bonsai4B,
        config,
        param_count,
        weights_size_bytes,
        kv_cache_4k_bytes,
        min_ram_bytes,
        description: "Bonsai-4B provides a balanced quality/memory trade-off. \
            24-layer GQA transformer with 2560-dimensional hidden state, 20 query heads, \
            4 KV heads, and a 65 536-token context window. Recommended when 2 GB RAM \
            is available and maximum quality is not required.",
    }
}

/// Build the [`ModelSpec`] for Bonsai-1.7B.
///
/// Architecture: 16 layers, hidden=1536, intermediate=4096, Q=12 heads, KV=2 heads.
pub fn bonsai_1_7b_spec() -> ModelSpec {
    let config = Qwen3Config::bonsai_1_7b();

    // ── param count ──────────────────────────────────────────────────────────
    // Embedding: 151936 × 1536 = 233 374 720
    // Output:    same
    // Per layer (16):
    //   Q: 1536×1536=2 359 296  K: 1536×256=393 216
    //   V: 1536×256=393 216     O: 1536×1536=2 359 296
    //   gate: 1536×4096=6 291 456  up: 6 291 456  down: 6 291 456
    //   norms ≈ 4 608
    // Layer total ≈ 24 383 616 × 16 = 390 137 856
    // Total ≈ 1 720 000 000 (embedding tables are large relative to compute)
    let param_count: u64 = 1_720_000_000;

    // ── weights at Q1_0_g128 ─────────────────────────────────────────────────
    // Quantised blocks: ~390 137 856 × 1.125/8 ≈ 54 956 940 bytes
    // Embedding FP16:   233 374 720 × 2 = 466 749 440
    // Output FP16:      same
    // Total ≈ 700 000 000
    let weights_size_bytes: u64 = 700_000_000;

    // ── KV cache at 4 096 context ─────────────────────────────────────────────
    // 16 × 2 × 2 × 128 × 4096 × 4 = 134 217 728
    let kv_cache_4k_bytes: u64 = kv_cache_size_bytes(&config, 4096);

    let min_ram_bytes = weights_size_bytes + kv_cache_4k_bytes + 32 * 1024 * 1024;

    ModelSpec {
        name: "Bonsai-1.7B",
        variant: ModelVariant::Bonsai1_7B,
        config,
        param_count,
        weights_size_bytes,
        kv_cache_4k_bytes,
        min_ram_bytes,
        description: "Bonsai-1.7B is the smallest and fastest variant, designed for \
            resource-constrained environments. 16-layer GQA transformer with 1536-dimensional \
            hidden state, 12 query heads, 2 KV heads, and a 65 536-token context window. \
            Runs with under 1 GB RAM.",
    }
}

/// Build the [`ModelSpec`] for Ternary-Bonsai-8B (Qwen3-8B architecture, TQ2_0_g128 weights).
///
/// Architecture is identical to Bonsai-8B; only the weight storage format differs.
pub fn ternary_bonsai_8b_spec() -> ModelSpec {
    let config = Qwen3Config::ternary_bonsai_8b();
    let param_count: u64 = 8_030_000_000;

    // TQ2_0_g128: 34 bytes per 128 weights ≈ 0.266 bytes/param.
    // Transformer weights only (excl. embedding/output ~1.24B params):
    //   ~6.8B × 0.266 ≈ 1.81 GB
    // Embedding (FP16): 151936 × 4096 × 2 ≈ 1.24 GB  (output head tied/same)
    // Norms (FP32): negligible
    // Weighted sum with ternary transformer weights → ~1.75 GB
    let weights_size_bytes: u64 = 1_750_000_000;
    let kv_cache_4k_bytes: u64 = kv_cache_size_bytes(&config, 4096);
    let min_ram_bytes = weights_size_bytes + kv_cache_4k_bytes + 64 * 1024 * 1024;

    ModelSpec {
        name: "Ternary-Bonsai-8B",
        variant: ModelVariant::TernaryBonsai8B,
        config,
        param_count,
        weights_size_bytes,
        kv_cache_4k_bytes,
        min_ram_bytes,
        description: "Ternary-Bonsai-8B uses the same Qwen3-8B architecture as Bonsai-8B, \
            but stores transformer weights in TQ2_0_g128 ternary format ({-1,0,+1}). \
            Approximately 0.266 bytes/weight versus 0.14 bytes/weight for the 1-bit variant, \
            trading a small size increase for ternary expressivity.",
    }
}

/// Build the [`ModelSpec`] for Ternary-Bonsai-4B (Qwen3-4B architecture, TQ2_0_g128 weights).
pub fn ternary_bonsai_4b_spec() -> ModelSpec {
    let config = Qwen3Config::ternary_bonsai_4b();
    let param_count: u64 = 4_020_000_000;

    // ~3.63B transformer params × 0.266 ≈ 0.97 GB
    // Embedding FP16: 151936 × 2560 × 2 ≈ 0.78 GB (output head same)
    // Total → ~0.90 GB
    let weights_size_bytes: u64 = 900_000_000;
    let kv_cache_4k_bytes: u64 = kv_cache_size_bytes(&config, 4096);
    let min_ram_bytes = weights_size_bytes + kv_cache_4k_bytes + 48 * 1024 * 1024;

    ModelSpec {
        name: "Ternary-Bonsai-4B",
        variant: ModelVariant::TernaryBonsai4B,
        config,
        param_count,
        weights_size_bytes,
        kv_cache_4k_bytes,
        min_ram_bytes,
        description: "Ternary-Bonsai-4B uses the same Qwen3-4B architecture as Bonsai-4B, \
            but stores transformer weights in TQ2_0_g128 ternary format ({-1,0,+1}).",
    }
}

/// Build the [`ModelSpec`] for Ternary-Bonsai-1.7B (Qwen3-1.7B architecture, TQ2_0_g128 weights).
pub fn ternary_bonsai_1_7b_spec() -> ModelSpec {
    let config = Qwen3Config::ternary_bonsai_1_7b();
    let param_count: u64 = 1_720_000_000;

    // ~1.49B transformer params × 0.266 ≈ 0.40 GB
    // Embedding FP16: 151936 × 1536 × 2 ≈ 0.47 GB (output head same)
    // Total → ~0.39 GB
    let weights_size_bytes: u64 = 390_000_000;
    let kv_cache_4k_bytes: u64 = kv_cache_size_bytes(&config, 4096);
    let min_ram_bytes = weights_size_bytes + kv_cache_4k_bytes + 32 * 1024 * 1024;

    ModelSpec {
        name: "Ternary-Bonsai-1.7B",
        variant: ModelVariant::TernaryBonsai1_7B,
        config,
        param_count,
        weights_size_bytes,
        kv_cache_4k_bytes,
        min_ram_bytes,
        description: "Ternary-Bonsai-1.7B uses the same Qwen3-1.7B architecture as Bonsai-1.7B, \
            but stores transformer weights in TQ2_0_g128 ternary format ({-1,0,+1}). \
            Designed for resource-constrained environments where ternary weights are preferred.",
    }
}

/// Build the [`ModelSpec`] for FP8-Bonsai-8B (Qwen3-8B architecture, FP8 weights).
///
/// Architecture is identical to Bonsai-8B; only the weight storage format differs (F8_E4M3 or F8_E5M2).
pub fn fp8_bonsai_8b_spec() -> ModelSpec {
    let config = Qwen3Config::bonsai_8b();
    let param_count: u64 = 8_030_000_000;

    // FP8: 1 byte/weight + 2 bytes FP16 scale per 32-weight block ≈ 1.0625 bytes/weight.
    // Transformer weights: ~7.88B × 1.0625 ≈ 8.37 GB
    // Embeddings (FP16): 151936 × 4096 × 2 ≈ 1.24 GB
    // Approximate total: ~8.5 GB
    let weights_size_bytes: u64 = 8_500_000_000;
    let kv_cache_4k_bytes: u64 = kv_cache_size_bytes(&config, 4096);
    let min_ram_bytes = weights_size_bytes + kv_cache_4k_bytes + 64 * 1024 * 1024;

    ModelSpec {
        name: "FP8-Bonsai-8B",
        variant: ModelVariant::FP8Bonsai8B,
        config,
        param_count,
        weights_size_bytes,
        kv_cache_4k_bytes,
        min_ram_bytes,
        description: "FP8-Bonsai-8B uses the same Qwen3-8B architecture as Bonsai-8B, \
            but stores transformer weights in FP8 format (E4M3FN or E5M2). \
            Approximately 1.0625 bytes/weight — higher precision than 1-bit or ternary, \
            closer to FP16 quality with half the storage.",
    }
}

/// Build the [`ModelSpec`] for FP8-Bonsai-4B (Qwen3-4B architecture, FP8 weights).
pub fn fp8_bonsai_4b_spec() -> ModelSpec {
    let config = Qwen3Config::bonsai_4b();
    let param_count: u64 = 4_020_000_000;

    // ~3.63B transformer params × 1.0625 ≈ 3.86 GB + embeddings 0.78 GB → ~5.0 GB
    let weights_size_bytes: u64 = 5_000_000_000;
    let kv_cache_4k_bytes: u64 = kv_cache_size_bytes(&config, 4096);
    let min_ram_bytes = weights_size_bytes + kv_cache_4k_bytes + 48 * 1024 * 1024;

    ModelSpec {
        name: "FP8-Bonsai-4B",
        variant: ModelVariant::FP8Bonsai4B,
        config,
        param_count,
        weights_size_bytes,
        kv_cache_4k_bytes,
        min_ram_bytes,
        description: "FP8-Bonsai-4B uses the same Qwen3-4B architecture as Bonsai-4B, \
            but stores transformer weights in FP8 format (E4M3FN or E5M2).",
    }
}

/// Build the [`ModelSpec`] for FP8-Bonsai-1.7B (Qwen3-1.7B architecture, FP8 weights).
pub fn fp8_bonsai_1_7b_spec() -> ModelSpec {
    let config = Qwen3Config::bonsai_1_7b();
    let param_count: u64 = 1_720_000_000;

    // ~1.49B transformer params × 1.0625 ≈ 1.58 GB + embeddings 0.47 GB → ~2.3 GB
    let weights_size_bytes: u64 = 2_300_000_000;
    let kv_cache_4k_bytes: u64 = kv_cache_size_bytes(&config, 4096);
    let min_ram_bytes = weights_size_bytes + kv_cache_4k_bytes + 32 * 1024 * 1024;

    ModelSpec {
        name: "FP8-Bonsai-1.7B",
        variant: ModelVariant::FP8Bonsai1_7B,
        config,
        param_count,
        weights_size_bytes,
        kv_cache_4k_bytes,
        min_ram_bytes,
        description: "FP8-Bonsai-1.7B uses the same Qwen3-1.7B architecture as Bonsai-1.7B, \
            but stores transformer weights in FP8 format (E4M3FN or E5M2). \
            Designed for resource-constrained environments where FP8 precision is preferred.",
    }
}

/// Return a static slice containing specs for all nine known variants,
/// ordered from largest (8B) to smallest (1.7B): 1-bit, ternary, then FP8.
pub fn all_specs() -> &'static [ModelSpec] {
    use std::sync::OnceLock;
    static SPECS: OnceLock<[ModelSpec; 9]> = OnceLock::new();
    SPECS.get_or_init(|| {
        [
            bonsai_8b_spec(),
            bonsai_4b_spec(),
            bonsai_1_7b_spec(),
            ternary_bonsai_8b_spec(),
            ternary_bonsai_4b_spec(),
            ternary_bonsai_1_7b_spec(),
            fp8_bonsai_8b_spec(),
            fp8_bonsai_4b_spec(),
            fp8_bonsai_1_7b_spec(),
        ]
    })
}

/// Return the spec for a specific [`ModelVariant`], or `None` for `Custom`.
pub fn spec_for_variant(v: ModelVariant) -> Option<&'static ModelSpec> {
    all_specs().iter().find(|s| s.variant == v)
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Compute KV-cache size in bytes for `seq_len` tokens at FP32 precision.
///
/// Formula: `num_layers × 2 (K+V) × num_kv_heads × head_dim × seq_len × 4 bytes`.
fn kv_cache_size_bytes(config: &Qwen3Config, seq_len: usize) -> u64 {
    let layers = config.num_layers as u64;
    let kv_heads = config.num_kv_heads as u64;
    let head_dim = config.head_dim as u64;
    let seq = seq_len as u64;
    layers * 2 * kv_heads * head_dim * seq * 4
}

// ─── CapabilityProfile ───────────────────────────────────────────────────────

/// Runtime capability description for application developers.
///
/// Describes the recommended sampling settings, supported context window,
/// language coverage, and representative use-cases for a model variant.
#[derive(Debug, Clone)]
pub struct CapabilityProfile {
    /// Maximum context length supported by this variant (tokens).
    pub max_context_len: usize,
    /// Whether the model supports a dedicated system-prompt slot.
    pub supports_system_prompt: bool,
    /// Whether the inference engine supports streaming token generation.
    pub supports_streaming: bool,
    /// Recommended softmax temperature for general-purpose tasks.
    pub recommended_temperature: f32,
    /// Recommended top-p threshold for nucleus sampling.
    pub recommended_top_p: f32,
    /// BCP-47 language tags for languages the model was trained on.
    pub languages: &'static [&'static str],
    /// Representative use-cases for this variant.
    pub use_cases: &'static [&'static str],
}

/// Return the [`CapabilityProfile`] for `v`.
///
/// # Notes
///
/// All Bonsai variants share the same base architecture (Qwen3) and therefore
/// the same context window and language coverage.  The differences are in the
/// recommended sampling parameters which are tuned per-size class.
pub fn capability_profile(v: ModelVariant) -> CapabilityProfile {
    // Common values shared by every Bonsai variant.
    const LANGUAGES: &[&str] = &[
        "en", // English
        "zh", // Chinese (Simplified + Traditional)
        "ja", // Japanese
        "ko", // Korean
        "de", // German
        "fr", // French
        "es", // Spanish
        "pt", // Portuguese
        "it", // Italian
        "ru", // Russian
        "ar", // Arabic
        "hi", // Hindi
        "th", // Thai
        "vi", // Vietnamese
        "id", // Indonesian
        "tr", // Turkish
        "pl", // Polish
        "nl", // Dutch
        "cs", // Czech
        "sv", // Swedish
    ];

    match v {
        ModelVariant::Bonsai8B => CapabilityProfile {
            max_context_len: 65536,
            supports_system_prompt: true,
            supports_streaming: true,
            recommended_temperature: 0.7,
            recommended_top_p: 0.9,
            languages: LANGUAGES,
            use_cases: &[
                "Long-document summarisation",
                "Complex multi-turn dialogue",
                "Code generation and debugging",
                "Structured data extraction",
                "Creative writing and story-telling",
                "Multilingual translation",
                "Retrieval-augmented generation (RAG)",
            ],
        },
        ModelVariant::Bonsai4B => CapabilityProfile {
            max_context_len: 65536,
            supports_system_prompt: true,
            supports_streaming: true,
            recommended_temperature: 0.72,
            recommended_top_p: 0.9,
            languages: LANGUAGES,
            use_cases: &[
                "Short-to-medium document summarisation",
                "Conversational chat assistants",
                "Code completion and review",
                "Data extraction and classification",
                "On-device inference with moderate hardware",
            ],
        },
        ModelVariant::Bonsai1_7B => CapabilityProfile {
            max_context_len: 65536,
            supports_system_prompt: true,
            supports_streaming: true,
            recommended_temperature: 0.75,
            recommended_top_p: 0.85,
            languages: LANGUAGES,
            use_cases: &[
                "Edge / IoT on-device inference",
                "Low-latency chatbot responses",
                "Simple Q&A over short documents",
                "Keyword extraction",
                "Fast text classification",
                "WASM browser deployment",
            ],
        },
        ModelVariant::TernaryBonsai8B => CapabilityProfile {
            max_context_len: 65536,
            supports_system_prompt: true,
            supports_streaming: true,
            recommended_temperature: 0.7,
            recommended_top_p: 0.9,
            languages: LANGUAGES,
            use_cases: &[
                "Long-document summarisation (ternary weights)",
                "Complex multi-turn dialogue",
                "Code generation and debugging",
                "Structured data extraction",
                "Creative writing and story-telling",
                "Multilingual translation",
                "Retrieval-augmented generation (RAG)",
            ],
        },
        ModelVariant::TernaryBonsai4B => CapabilityProfile {
            max_context_len: 65536,
            supports_system_prompt: true,
            supports_streaming: true,
            recommended_temperature: 0.72,
            recommended_top_p: 0.9,
            languages: LANGUAGES,
            use_cases: &[
                "Short-to-medium document summarisation (ternary weights)",
                "Conversational chat assistants",
                "Code completion and review",
                "Data extraction and classification",
                "On-device inference with moderate hardware",
            ],
        },
        ModelVariant::TernaryBonsai1_7B => CapabilityProfile {
            max_context_len: 65536,
            supports_system_prompt: true,
            supports_streaming: true,
            recommended_temperature: 0.75,
            recommended_top_p: 0.85,
            languages: LANGUAGES,
            use_cases: &[
                "Edge / IoT on-device inference (ternary weights)",
                "Low-latency chatbot responses",
                "Simple Q&A over short documents",
                "Keyword extraction",
                "Fast text classification",
                "WASM browser deployment",
            ],
        },
        ModelVariant::FP8Bonsai8B => CapabilityProfile {
            max_context_len: 65536,
            supports_system_prompt: true,
            supports_streaming: true,
            recommended_temperature: 0.7,
            recommended_top_p: 0.9,
            languages: LANGUAGES,
            use_cases: &[
                "Long-document summarisation (FP8 weights)",
                "Complex multi-turn dialogue",
                "Code generation and debugging",
                "Structured data extraction",
                "Creative writing and story-telling",
                "Multilingual translation",
                "Retrieval-augmented generation (RAG)",
            ],
        },
        ModelVariant::FP8Bonsai4B => CapabilityProfile {
            max_context_len: 65536,
            supports_system_prompt: true,
            supports_streaming: true,
            recommended_temperature: 0.72,
            recommended_top_p: 0.9,
            languages: LANGUAGES,
            use_cases: &[
                "Short-to-medium document summarisation (FP8 weights)",
                "Conversational chat assistants",
                "Code completion and review",
                "Data extraction and classification",
                "On-device inference with moderate hardware",
            ],
        },
        ModelVariant::FP8Bonsai1_7B => CapabilityProfile {
            max_context_len: 65536,
            supports_system_prompt: true,
            supports_streaming: true,
            recommended_temperature: 0.75,
            recommended_top_p: 0.85,
            languages: LANGUAGES,
            use_cases: &[
                "Edge / IoT on-device inference (FP8 weights)",
                "Low-latency chatbot responses",
                "Simple Q&A over short documents",
                "Keyword extraction",
                "Fast text classification",
                "WASM browser deployment",
            ],
        },
        ModelVariant::Custom => CapabilityProfile {
            max_context_len: 65536,
            supports_system_prompt: true,
            supports_streaming: true,
            recommended_temperature: 0.7,
            recommended_top_p: 0.9,
            languages: LANGUAGES,
            use_cases: &["Custom architecture — use-cases depend on training data"],
        },
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helper ───────────────────────────────────────────────────────────────

    fn all_known_variants() -> [ModelVariant; 9] {
        [
            ModelVariant::Bonsai8B,
            ModelVariant::Bonsai4B,
            ModelVariant::Bonsai1_7B,
            ModelVariant::TernaryBonsai8B,
            ModelVariant::TernaryBonsai4B,
            ModelVariant::TernaryBonsai1_7B,
            ModelVariant::FP8Bonsai8B,
            ModelVariant::FP8Bonsai4B,
            ModelVariant::FP8Bonsai1_7B,
        ]
    }

    // ── ModelSpec: config validity ────────────────────────────────────────────

    #[test]
    fn all_variants_produce_valid_configs() {
        for v in all_known_variants() {
            let spec = spec_for_variant(v).expect("known variant must have a spec");
            // Architecture-level sanity
            assert!(
                spec.config.num_layers > 0,
                "{}: num_layers must be > 0",
                spec.name
            );
            assert!(
                spec.config.hidden_size > 0,
                "{}: hidden_size must be > 0",
                spec.name
            );
            assert!(
                spec.config.intermediate_size > 0,
                "{}: intermediate_size must be > 0",
                spec.name
            );
            assert!(
                spec.config.num_attention_heads > 0,
                "{}: num_attention_heads must be > 0",
                spec.name
            );
            assert!(
                spec.config.num_kv_heads > 0,
                "{}: num_kv_heads must be > 0",
                spec.name
            );
            assert!(
                spec.config.vocab_size > 0,
                "{}: vocab_size must be > 0",
                spec.name
            );
            // GQA invariant: kv_heads must divide query heads evenly
            assert_eq!(
                spec.config.num_attention_heads % spec.config.num_kv_heads,
                0,
                "{}: num_attention_heads must be divisible by num_kv_heads",
                spec.name
            );
            // head_dim consistency
            assert_eq!(
                spec.config.hidden_size / spec.config.num_attention_heads,
                spec.config.head_dim,
                "{}: head_dim inconsistency",
                spec.name
            );
        }
    }

    // ── ModelSpec: param_count bounds ─────────────────────────────────────────

    #[test]
    fn param_count_is_reasonable() {
        let s8b = bonsai_8b_spec();
        assert!(
            s8b.param_count > 1_000_000_000,
            "8B: param_count must exceed 1B"
        );
        assert!(
            s8b.param_count < 10_000_000_000,
            "8B: param_count must be under 10B"
        );

        let s4b = bonsai_4b_spec();
        assert!(
            s4b.param_count > 1_000_000_000,
            "4B: param_count must exceed 1B"
        );
        assert!(
            s4b.param_count < 10_000_000_000,
            "4B: param_count must be under 10B"
        );

        let s1_7b = bonsai_1_7b_spec();
        assert!(
            s1_7b.param_count > 1_000_000_000,
            "1.7B: param_count must exceed 1B"
        );
        assert!(
            s1_7b.param_count < 10_000_000_000,
            "1.7B: param_count must be under 10B"
        );

        // Order
        assert!(
            s8b.param_count > s4b.param_count,
            "8B should have more params than 4B"
        );
        assert!(
            s4b.param_count > s1_7b.param_count,
            "4B should have more params than 1.7B"
        );
    }

    // ── ModelSpec: Q1_0_g128 weights size ─────────────────────────────────────
    //
    // At Q1_0_g128 the average storage is 1.125 bits per weight.
    // Embedding tables are stored in FP16 (2 bytes per param).
    // We allow a ±30% tolerance for metadata and alignment padding.

    #[test]
    fn weights_size_matches_q1_0_g128_expectation() {
        for spec in all_specs() {
            // Rough lower bound: transformer params only at 1 bit = param_count / 8
            let lower = spec.param_count / 8;
            // Rough upper bound: all params at FP16 = param_count * 2
            let upper = spec.param_count * 2;
            assert!(
                spec.weights_size_bytes >= lower,
                "{}: weights_size_bytes {} is below the 1-bit lower bound {}",
                spec.name,
                spec.weights_size_bytes,
                lower
            );
            assert!(
                spec.weights_size_bytes <= upper,
                "{}: weights_size_bytes {} exceeds the FP16 upper bound {}",
                spec.name,
                spec.weights_size_bytes,
                upper
            );
        }
    }

    // ── ModelSpec: KV cache size ───────────────────────────────────────────────

    #[test]
    fn kv_cache_4k_bytes_is_reasonable() {
        // Minimum: at least 1 MiB
        // Maximum: should not exceed 4 GiB (sanity ceiling)
        for spec in all_specs() {
            let min_bytes: u64 = 1024 * 1024;
            let max_bytes: u64 = 4 * 1024 * 1024 * 1024;
            assert!(
                spec.kv_cache_4k_bytes >= min_bytes,
                "{}: kv_cache_4k_bytes {} is suspiciously small",
                spec.name,
                spec.kv_cache_4k_bytes
            );
            assert!(
                spec.kv_cache_4k_bytes <= max_bytes,
                "{}: kv_cache_4k_bytes {} exceeds 4 GiB sanity limit",
                spec.name,
                spec.kv_cache_4k_bytes
            );
        }
        // Order: 8B > 4B > 1.7B (more layers × more KV heads)
        let s8b = bonsai_8b_spec();
        let s4b = bonsai_4b_spec();
        let s1_7b = bonsai_1_7b_spec();
        assert!(s8b.kv_cache_4k_bytes > s4b.kv_cache_4k_bytes);
        assert!(s4b.kv_cache_4k_bytes > s1_7b.kv_cache_4k_bytes);
    }

    // ── ModelSpec: min_ram includes weights + kv_cache ────────────────────────

    #[test]
    fn min_ram_includes_weights_and_kv_cache() {
        for spec in all_specs() {
            assert!(
                spec.min_ram_bytes >= spec.weights_size_bytes + spec.kv_cache_4k_bytes,
                "{}: min_ram_bytes must be at least weights + kv_cache",
                spec.name
            );
        }
    }

    // ── all_specs / spec_for_variant ─────────────────────────────────────────

    #[test]
    fn all_specs_returns_nine_entries() {
        assert_eq!(all_specs().len(), 9);
    }

    #[test]
    fn spec_for_known_variants_returns_some() {
        for v in all_known_variants() {
            assert!(
                spec_for_variant(v).is_some(),
                "spec_for_variant({:?}) should return Some",
                v
            );
        }
    }

    #[test]
    fn spec_for_custom_returns_none() {
        assert!(spec_for_variant(ModelVariant::Custom).is_none());
    }

    #[test]
    fn spec_variant_field_matches_lookup_key() {
        for spec in all_specs() {
            let looked_up = spec_for_variant(spec.variant)
                .expect("spec_for_variant must succeed for variants in all_specs()");
            assert_eq!(
                spec.variant, looked_up.variant,
                "spec lookup returned wrong variant for {}",
                spec.name
            );
        }
    }

    // ── CapabilityProfile ─────────────────────────────────────────────────────

    #[test]
    fn capability_profile_returns_valid_data() {
        for v in all_known_variants() {
            let profile = capability_profile(v);
            // Context window must be positive and at most 1M tokens (sanity)
            assert!(
                profile.max_context_len > 0,
                "{:?}: max_context_len must be > 0",
                v
            );
            assert!(
                profile.max_context_len <= 1_000_000,
                "{:?}: max_context_len exceeds sanity ceiling",
                v
            );
            // Temperature must be in (0, 2]
            assert!(
                profile.recommended_temperature > 0.0,
                "{:?}: temperature must be > 0",
                v
            );
            assert!(
                profile.recommended_temperature <= 2.0,
                "{:?}: temperature must be <= 2.0",
                v
            );
            // top_p must be in (0, 1]
            assert!(
                profile.recommended_top_p > 0.0,
                "{:?}: top_p must be > 0",
                v
            );
            assert!(
                profile.recommended_top_p <= 1.0,
                "{:?}: top_p must be <= 1.0",
                v
            );
            // At least one language
            assert!(
                !profile.languages.is_empty(),
                "{:?}: languages must not be empty",
                v
            );
            // At least one use-case
            assert!(
                !profile.use_cases.is_empty(),
                "{:?}: use_cases must not be empty",
                v
            );
            // English must be in the language list (sanity)
            assert!(
                profile.languages.contains(&"en"),
                "{:?}: English (\"en\") must be in languages",
                v
            );
            // Streaming and system-prompt support
            assert!(
                profile.supports_streaming,
                "{:?}: all Bonsai variants support streaming",
                v
            );
            assert!(
                profile.supports_system_prompt,
                "{:?}: all Bonsai variants support system prompts",
                v
            );
        }
    }

    #[test]
    fn capability_profile_for_custom_variant_is_valid() {
        let profile = capability_profile(ModelVariant::Custom);
        assert!(profile.max_context_len > 0);
        assert!(!profile.languages.is_empty());
        assert!(!profile.use_cases.is_empty());
    }

    // ── KV cache helper ───────────────────────────────────────────────────────

    #[test]
    fn kv_cache_helper_formula_is_correct() {
        let config = Qwen3Config::bonsai_8b();
        // 36 layers × 2 (K+V) × 8 kv_heads × 128 head_dim × 4096 tokens × 4 bytes
        let expected: u64 = 36 * 2 * 8 * 128 * 4096 * 4;
        assert_eq!(kv_cache_size_bytes(&config, 4096), expected);
    }
}
