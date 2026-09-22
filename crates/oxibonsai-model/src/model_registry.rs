//! Multi-model support: auto-detect Bonsai model variant from GGUF metadata.
//!
//! The model registry provides automatic detection of model architecture
//! variants (8B, 4B, 1.7B) based on configuration parameters like
//! layer count and hidden dimension size.

use oxibonsai_core::config::Qwen3Config;

/// Best-effort `Qwen3Config` projection of the real `qwen35` Bonsai 2 27B
/// header (design doc Appendix A.4 / A.1) — every scalar field
/// `Qwen3Config` has a slot for, filled with the real value. See
/// [`ModelVariant::default_config`]'s doc comment for what this
/// deliberately cannot represent (M-RoPE sections, SSM/Gated-DeltaNet
/// parameters, the Hadamard fold) and why that is fine for this method's
/// callers (capability reporting / size estimation, not forward-pass
/// construction).
fn qwen35_27b_scalar_config() -> Qwen3Config {
    Qwen3Config {
        hidden_size: 5120,
        intermediate_size: 17408,
        num_layers: 64,
        num_attention_heads: 24,
        num_kv_heads: 4,
        head_dim: 256,
        value_length: 256,
        vocab_size: 248320,
        max_context_length: 262144,
        rms_norm_eps: 1e-6,
        rope_freq_base: 1.0e7,
        // `qwen35` uses 3-axis M-RoPE with `rope.dimension_sections`, which
        // has no `RopeScaling` representation; `None` is the honest "this
        // field cannot express what the real model does" answer, not a
        // claim that the model runs unscaled.
        rope_scaling: oxibonsai_core::config::RopeScaling::None,
        sliding_window: None,
        architecture: "qwen35".to_string(),
        model_name: "Bonsai-2-27B".to_string(),
    }
}

/// Known Bonsai model variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelVariant {
    /// Bonsai-8B (Qwen3-8B architecture): 36 layers, hidden=4096
    Bonsai8B,
    /// Bonsai-4B: 24 layers, hidden=2560
    Bonsai4B,
    /// Bonsai-1.7B: 28 layers, hidden=2048, intermediate=6144, 16 heads, 8 kv heads
    Bonsai1_7B,
    /// Ternary-Bonsai-8B: same Qwen3-8B architecture, {-1,0,+1} weights (TQ2_0_g128).
    TernaryBonsai8B,
    /// Ternary-Bonsai-4B: same Qwen3-4B architecture, {-1,0,+1} weights (TQ2_0_g128).
    TernaryBonsai4B,
    /// Ternary-Bonsai-1.7B: same Qwen3-1.7B architecture, {-1,0,+1} weights (TQ2_0_g128).
    TernaryBonsai1_7B,
    /// FP8-Bonsai-8B: same Qwen3-8B architecture, FP8 weights (F8_E4M3 or F8_E5M2).
    FP8Bonsai8B,
    /// FP8-Bonsai-4B: same Qwen3-4B architecture, FP8 weights.
    FP8Bonsai4B,
    /// FP8-Bonsai-1.7B: same Qwen3-1.7B architecture, FP8 weights.
    FP8Bonsai1_7B,
    /// Gen-1 Bonsai 27B: `qwen35` hybrid architecture (64 layers, hidden
    /// 5120), **no** Hadamard fold. Covers all three no-hadamard 27B files
    /// regardless of bit-width (`Ternary-Bonsai-27B-Q2_0`,
    /// `Ternary-Bonsai-27B-PQ2_0`, `Bonsai-27B-Q1_0`) — one ModelVariant per
    /// *generation*, not per quant format, matching design §3.9.
    Bonsai27B,
    /// Bonsai 2 27B (`qwen35` hybrid, Hadamard-folded): PrismML `PQ2_0`
    /// ternary codec (ggml id 142) — `Ternary-Bonsai-2-27B-PQ2_0.gguf`.
    TernaryBonsai227bPq2,
    /// Bonsai 2 27B (`qwen35` hybrid, Hadamard-folded): PrismML `PTQ1_0`
    /// 1.75-bit codec (ggml id 143) — `Ternary-Bonsai-2-27B-PTQ1_0.gguf`.
    TernaryBonsai227bPtq1,
    /// Bonsai 2 27B (`qwen35` hybrid, Hadamard-folded): mainline group-64
    /// `Q2_0` codec (ggml id 42, resolved) —
    /// `Ternary-Bonsai-2-27B-Q2_0-prism-fork-required.gguf`.
    TernaryBonsai227bQ2g64,
    /// Bonsai 2 27B vision projector (`clip` architecture,
    /// `qwen3vl_merger`, 27 ViT blocks, Q8_0) —
    /// `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf`. Not a language model on its
    /// own; paired with one of the four `qwen35` variants above.
    Bonsai227bMmproj,
    /// Custom or unrecognized architecture
    Custom,
}

impl ModelVariant {
    /// Auto-detect variant from model configuration.
    ///
    /// Matches on the combination of `num_layers` and `hidden_size`
    /// to identify known architectures.
    pub fn from_config(config: &Qwen3Config) -> Self {
        match (config.num_layers, config.hidden_size) {
            (36, 4096) => ModelVariant::Bonsai8B,
            (24, 2560) => ModelVariant::Bonsai4B,
            (28, 2048) => ModelVariant::Bonsai1_7B,
            _ => ModelVariant::Custom,
        }
    }

    /// Detect model variant from config + sample tensor type (for ternary vs 1-bit disambiguation).
    ///
    /// Architecture match is identical to `from_config`, but if `sample_tensor_type.is_ternary()`,
    /// the result is upgraded to the ternary sibling variant.
    pub fn from_config_and_sample_tensor_type(
        config: &Qwen3Config,
        sample_tensor_type: oxibonsai_core::GgufTensorType,
    ) -> Self {
        let base = Self::from_config(config);
        if sample_tensor_type.is_ternary() {
            match base {
                Self::Bonsai8B => Self::TernaryBonsai8B,
                Self::Bonsai4B => Self::TernaryBonsai4B,
                Self::Bonsai1_7B => Self::TernaryBonsai1_7B,
                other => other, // Custom or already-ternary → unchanged
            }
        } else if sample_tensor_type.is_fp8() {
            match base {
                Self::Bonsai8B => Self::FP8Bonsai8B,
                Self::Bonsai4B => Self::FP8Bonsai4B,
                Self::Bonsai1_7B => Self::FP8Bonsai1_7B,
                other => other, // Custom or already-fp8 → unchanged
            }
        } else {
            base
        }
    }

    /// [`Self::from_config_and_sample_tensor_type`], additionally trying the
    /// Bonsai 2 27B / `qwen35` family first (M-14 / cli-01 / cli-16).
    ///
    /// [`Self::from_config_and_sample_tensor_type`] alone cannot recognise a
    /// real 27B file: `Self::from_config`'s `(num_layers, hidden_size)`
    /// table has no `(64, 5120)` arm, so it falls to [`Self::Custom`] —
    /// whose [`Self::param_count`]/[`Self::expected_model_size_bytes`] are
    /// defined as exactly `0` — even though [`Self::detect_qwen35_27b`]
    /// knows exactly which variant it is. This is the production entry
    /// point that closes that gap: it tries [`Self::detect_qwen35_27b`]
    /// first (never guessing from the raw ambiguous ggml id 42 -- pass
    /// `resolved_type` from
    /// [`oxibonsai_core::gguf::quant_resolve::resolve_type_42_with_sample`]-style
    /// resolution, never a tensor's raw parse-time type) and only falls
    /// through to the legacy 8B/4B/1.7B-family detection on `None`.
    ///
    /// `has_hadamard` is whether `prism.hadamard.version` (or any
    /// `prism.hadamard.*` key) is present in the file's metadata --
    /// [`Qwen3Config`] has no field for it, so it is a separate parameter
    /// rather than something this can derive from `config` alone.
    pub fn from_config_and_resolved_sample(
        config: &Qwen3Config,
        resolved_type: oxibonsai_core::GgufTensorType,
        has_hadamard: bool,
    ) -> Self {
        Self::detect_qwen35_27b(
            &config.architecture,
            config.num_layers as u64,
            resolved_type,
            has_hadamard,
        )
        .unwrap_or_else(|| Self::from_config_and_sample_tensor_type(config, resolved_type))
    }

    /// Detect a Bonsai 2 27B / `qwen35` variant (or its `clip` mmproj
    /// companion) from raw GGUF signals, **never** from the raw ambiguous
    /// ggml id 42 (M-14 / cli-01 / cli-16; design §3.9).
    ///
    /// - `architecture`: `general.architecture` (`"qwen35"` for the
    ///   language model, `"clip"` for the mmproj projector).
    /// - `block_count`: `<architecture>.block_count` for the language
    ///   model (64 for every real 27B file) or `clip.vision.block_count`
    ///   for the projector (27 for the real mmproj file).
    /// - `resolved_type`: a sample weight tensor's type **after**
    ///   [`oxibonsai_core::gguf::quant_resolve::resolve_type_42`]-style
    ///   disambiguation — ignored for `architecture == "clip"`.
    /// - `has_hadamard`: whether `prism.hadamard.version` (or any
    ///   `prism.hadamard.*` key) is present in the file's metadata.
    ///
    /// Returns `None` for a `(architecture, block_count)` this build does
    /// not recognise, or a `(resolved_type, has_hadamard)` combination none
    /// of the six real 27B files use — a caller must not guess a variant
    /// for those, matching this function's own contract of never guessing.
    ///
    /// The `Q2_0G128DFirst` arm is the legacy PrismML gen-1 reading of ggml
    /// id 42 (wire-identical to `PQ2_0`, per `dequant_any`'s comment in
    /// `weight_loaders.rs`) — `Ternary-Bonsai-27B-Q2_0.gguf` resolves to it,
    /// not to `PQ2_0` itself (that id is reserved for the true `PQ2_0`
    /// files, both of which declare `prism.hadamard.*`).
    pub fn detect_qwen35_27b(
        architecture: &str,
        block_count: u64,
        resolved_type: oxibonsai_core::GgufTensorType,
        has_hadamard: bool,
    ) -> Option<Self> {
        use oxibonsai_core::GgufTensorType;

        if architecture == "clip" {
            return (block_count == 27).then_some(Self::Bonsai227bMmproj);
        }
        if architecture != "qwen35" || block_count != 64 {
            return None;
        }
        match (resolved_type, has_hadamard) {
            (GgufTensorType::PQ2_0, true) => Some(Self::TernaryBonsai227bPq2),
            (GgufTensorType::PTQ1_0, true) => Some(Self::TernaryBonsai227bPtq1),
            (GgufTensorType::Q2_0G64, true) => Some(Self::TernaryBonsai227bQ2g64),
            // Gen-1 (no Hadamard fold): one ModelVariant regardless of
            // bit-width (design §3.9) — legacy g128 Q2_0
            // (`Ternary-Bonsai-27B-Q2_0.gguf`), PQ2_0
            // (`Ternary-Bonsai-27B-PQ2_0.gguf`), or 1-bit
            // (`Bonsai-27B-Q1_0.gguf`).
            (
                GgufTensorType::Q2_0G128DFirst | GgufTensorType::PQ2_0 | GgufTensorType::Q1_0_g128,
                false,
            ) => Some(Self::Bonsai27B),
            // Every other combination (e.g. PTQ1_0/Q2_0G64 without a
            // Hadamard fold) is not one of the six real files — refuse to
            // guess rather than misclassify a checkpoint this build has
            // never seen.
            _ => None,
        }
    }

    /// Get the default configuration for this variant.
    ///
    /// Returns the standard configuration for known variants.
    /// For `Custom`, returns the 8B configuration as a fallback.
    ///
    /// **The four `qwen35` (Bonsai 2 family) variants and the mmproj
    /// variant are `Qwen3Config`-incompatible** (B2-09): `qwen35` is a
    /// hybrid architecture with M-RoPE sections, SSM/Gated-DeltaNet
    /// parameters and a Hadamard fold that `Qwen3Config` has no fields for,
    /// and `Bonsai227bMmproj` is a `clip` vision tower, not a causal LM at
    /// all. This method still returns a best-effort `Qwen3Config` for the
    /// four `qwen35` variants (every scalar field `Qwen3Config` *can*
    /// represent, set to the real GGUF header value — Appendix A.4) so
    /// generic scalar-config consumers (capability reporting, parameter/size
    /// estimators) get real numbers instead of a zeroed `Custom` guess (the
    /// M-14 defect this package closes) — **it is not sufficient to
    /// construct a working hybrid forward pass**; that needs the real
    /// `qwen35`-specific config B2-10 introduces. `Bonsai227bMmproj` returns
    /// the 8B placeholder like `Custom`, since none of `Qwen3Config`'s
    /// fields have a sensible mapping for a ViT.
    pub fn default_config(&self) -> Qwen3Config {
        match self {
            ModelVariant::Bonsai8B => Qwen3Config::bonsai_8b(),
            ModelVariant::Bonsai4B => Qwen3Config::bonsai_4b(),
            ModelVariant::Bonsai1_7B => Qwen3Config::bonsai_1_7b(),
            ModelVariant::TernaryBonsai8B => Qwen3Config::ternary_bonsai_8b(),
            ModelVariant::TernaryBonsai4B => Qwen3Config::ternary_bonsai_4b(),
            ModelVariant::TernaryBonsai1_7B => Qwen3Config::ternary_bonsai_1_7b(),
            // FP8 variants share the same Qwen3 architecture as their 1-bit siblings.
            ModelVariant::FP8Bonsai8B => Qwen3Config::bonsai_8b(),
            ModelVariant::FP8Bonsai4B => Qwen3Config::bonsai_4b(),
            ModelVariant::FP8Bonsai1_7B => Qwen3Config::bonsai_1_7b(),
            ModelVariant::Bonsai27B
            | ModelVariant::TernaryBonsai227bPq2
            | ModelVariant::TernaryBonsai227bPtq1
            | ModelVariant::TernaryBonsai227bQ2g64 => qwen35_27b_scalar_config(),
            ModelVariant::Bonsai227bMmproj | ModelVariant::Custom => Qwen3Config::bonsai_8b(),
        }
    }

    /// Human-readable display name for this variant.
    pub fn name(&self) -> &'static str {
        match self {
            ModelVariant::Bonsai8B => "Bonsai-8B",
            ModelVariant::Bonsai4B => "Bonsai-4B",
            ModelVariant::Bonsai1_7B => "Bonsai-1.7B",
            ModelVariant::TernaryBonsai8B => "Ternary-Bonsai-8B",
            ModelVariant::TernaryBonsai4B => "Ternary-Bonsai-4B",
            ModelVariant::TernaryBonsai1_7B => "Ternary-Bonsai-1.7B",
            ModelVariant::FP8Bonsai8B => "FP8-Bonsai-8B",
            ModelVariant::FP8Bonsai4B => "FP8-Bonsai-4B",
            ModelVariant::FP8Bonsai1_7B => "FP8-Bonsai-1.7B",
            ModelVariant::Bonsai27B => "Bonsai-27B",
            ModelVariant::TernaryBonsai227bPq2 => "Ternary-Bonsai-2-27B-PQ2_0",
            ModelVariant::TernaryBonsai227bPtq1 => "Ternary-Bonsai-2-27B-PTQ1_0",
            ModelVariant::TernaryBonsai227bQ2g64 => "Ternary-Bonsai-2-27B-Q2_0",
            ModelVariant::Bonsai227bMmproj => "Bonsai-2-27B-mmproj",
            ModelVariant::Custom => "Custom",
        }
    }

    /// Approximate parameter count for this variant.
    ///
    /// Computed as: embedding + attention + ffn + norms + output head.
    /// For 1-bit models, each "parameter" is 1 bit + per-group scale.
    /// Ternary variants share the same architecture (and thus the same parameter count)
    /// as their 1-bit siblings; only the storage format differs.
    pub fn param_count(&self) -> u64 {
        match self {
            ModelVariant::Bonsai8B | ModelVariant::TernaryBonsai8B | ModelVariant::FP8Bonsai8B => {
                // Qwen3-8B (real GGUF header / Qwen3Config::bonsai_8b(),
                // corrected by M-34): ~8.03B parameters
                // Embedding: 151669 * 4096 = 621M
                // Per layer: Q(4096*4096) + K(4096*1024) + V(4096*1024) + O(4096*4096)
                //          + gate(4096*12288) + up(4096*12288) + down(12288*4096)
                //          + 2 norms(4096 each)
                // = 16M + 4M + 4M + 16M + 50.3M + 50.3M + 50.3M + 8K = ~191M per layer
                // 36 layers = ~6.88B
                // + embedding(621M) + output(621M) + final norm(4K)
                8_030_000_000
            }
            ModelVariant::Bonsai4B | ModelVariant::TernaryBonsai4B | ModelVariant::FP8Bonsai4B => {
                // 24 layers, hidden=2560, intermediate=6912
                // Per layer: Q(2560*2560) + K(2560*512) + V(2560*512) + O(2560*2560)
                //          + gate(2560*6912) + up(2560*6912) + down(6912*2560) + norms
                // Embedding: 151936 * 2560
                4_020_000_000
            }
            ModelVariant::Bonsai1_7B
            | ModelVariant::TernaryBonsai1_7B
            | ModelVariant::FP8Bonsai1_7B => {
                // 28 layers, hidden=2048, intermediate=6144, 16 heads, 8 kv heads
                1_720_000_000
            }
            ModelVariant::Bonsai27B
            | ModelVariant::TernaryBonsai227bPq2
            | ModelVariant::TernaryBonsai227bPtq1
            | ModelVariant::TernaryBonsai227bQ2g64 => {
                // qwen35 hybrid, 64 layers (16 full + 48 linear), hidden 5120
                // (design Appendix A.4). Embedding + output (both
                // 5120*248320, `output.weight` always present, not tied):
                //   2 * 1_271_398_400 = 2_542_796_800
                // Full layer: Q(5120*12288) + K(5120*1024) + V(5120*1024)
                //   + O(6144*5120) + ffn(gate/up/down, 5120*17408 each) + norms
                //   ~= 372_255_248; x16 full layers ~= 5_956_083_968
                // Linear layer: qkv(5120*10240) + gate(5120*6144)
                //   + ssm_alpha/beta(5120*48 each, tiny) + ssm_out(6144*5120)
                //   + ffn(gate/up/down) + norms ~= 383_273_712; x48 linear
                //   layers ~= 18_397_138_176
                // Grand total ~= 26_896_000_000 (matches the "27B" name)
                26_900_000_000
            }
            // Vision projector (`clip` architecture): ~630 MB at Q8_0
            // (~1 byte/param) puts the ViT + merger around 620M parameters —
            // an order of magnitude smaller than the language model it
            // pairs with, not a language model itself.
            ModelVariant::Bonsai227bMmproj => 620_000_000,
            ModelVariant::Custom => 0,
        }
    }

    /// Expected model file size in bytes for the quantized GGUF file.
    ///
    /// For 1-bit variants: ~1 bit per param + scale factors + FP16 embeddings.
    /// For ternary variants: TQ2_0_g128 uses 34 bytes per 128 weights ≈ 0.266 bytes/param.
    /// Embeddings and norms are typically stored in FP16 or FP32.
    pub fn expected_model_size_bytes(&self) -> u64 {
        match self {
            ModelVariant::Bonsai8B => {
                // ~8B params at 1 bit = ~1 GB for weights
                // + embeddings in FP16: 151669 * 4096 * 2 = ~1.24 GB (real
                // GGUF header shape corrected by M-34: vocab=151669)
                // + norms in FP32: ~0.01 GB
                // + metadata overhead
                // Total: ~2.2 GB
                2_200_000_000
            }
            ModelVariant::Bonsai4B => {
                // ~4B params at 1 bit = ~0.5 GB
                // + embeddings in FP16: 151936 * 2560 * 2 = ~0.78 GB
                // Total: ~1.3 GB
                1_300_000_000
            }
            ModelVariant::Bonsai1_7B => {
                // ~1.7B params at 1 bit = ~0.21 GB
                // + embeddings in FP16: 151669 * 2048 * 2 ≈ 0.62 GB (real
                // GGUF header shape corrected by M-34: hidden=2048, vocab=151669)
                // Total: ~0.7 GB
                700_000_000
            }
            ModelVariant::TernaryBonsai8B => {
                // TQ2_0_g128: 34 bytes per 128 weights ≈ 0.266 bytes/param
                // ~8.03B params × 0.266 ≈ ~2.13 GB minus embeddings sharing
                // Embeddings (FP16): 151669 * 4096 * 2 ≈ 1.24 GB — same as
                // 1-bit (real GGUF header shape corrected by M-34: vocab=151669)
                // Transformer weights only (excl. embedding/output ~1.24B params):
                //   ~6.8B × 0.266 ≈ 1.81 GB + embedding 1.24 GB → ~1.75 GB total
                // (embeddings/output stored in FP16 dominate less at ternary density)
                1_750_000_000
            }
            ModelVariant::TernaryBonsai4B => {
                // ~4.02B params, transformer weights ~3.63B × 0.266 ≈ 0.97 GB
                // + embeddings (FP16): 151936 * 2560 * 2 ≈ 0.78 GB → ~0.90 GB total
                900_000_000
            }
            ModelVariant::TernaryBonsai1_7B => {
                // ~1.72B params, transformer weights ~1.41B × 0.266 ≈ 0.37 GB
                // (real GGUF header shape corrected by M-34: 28 layers, hidden=2048)
                // + embeddings (FP16): 151669 * 2048 * 2 ≈ 0.62 GB → ~0.39 GB total
                390_000_000
            }
            ModelVariant::FP8Bonsai8B => {
                // FP8: 1 byte/weight + FP16 scale per 32-weight block ≈ 1.0625 bytes/weight
                // Transformer weights: ~7.88B × 1.0625 ≈ 8.37 GB — but embeddings in FP16
                // Embeddings (FP16): 151669 × 4096 × 2 ≈ 1.24 GB (real GGUF
                // header shape corrected by M-34: vocab=151669)
                // Rough total: ~8.5 GB (FP8 is closer to FP16 in size)
                8_500_000_000
            }
            ModelVariant::FP8Bonsai4B => {
                // Transformer: ~3.63B × 1.0625 ≈ 3.86 GB + embeddings 0.78 GB → ~5.0 GB
                5_000_000_000
            }
            ModelVariant::FP8Bonsai1_7B => {
                // Transformer: ~1.41B × 1.0625 ≈ 1.50 GB + embeddings 0.62 GB → ~2.12 GB
                // (real GGUF header shape corrected by M-34: 28 layers, hidden=2048,
                // vocab=151669)
                2_300_000_000
            }
            // Real on-disk file sizes (design doc Appendix A.4, measured
            // from the actual GGUF headers).
            ModelVariant::TernaryBonsai227bPq2 => 7_206_168_928,
            ModelVariant::TernaryBonsai227bPtq1 => 5_946_648_928,
            // No file-size measurement exists for the mainline group-64
            // Q2_0 codec specifically (only PQ2_0/PTQ1_0 were measured); its
            // bits/weight (18 B / 64 = 2.25) is close to PQ2_0's (34 B / 128
            // = 2.125), so this is PQ2_0's measured size scaled by that
            // ratio — an estimate, not a measurement.
            ModelVariant::TernaryBonsai227bQ2g64 => 7_650_000_000,
            // Gen-1 (no Hadamard fold) 27B: covers three files at three
            // different bit-widths (legacy g128 Q2_0, PQ2_0, Q1_0) under one
            // ModelVariant (design §3.9), so no single byte count is exact;
            // this picks the middle (ternary-density) file's ballpark.
            ModelVariant::Bonsai27B => 7_200_000_000,
            // Real on-disk mmproj file size (design doc Appendix A.4).
            ModelVariant::Bonsai227bMmproj => 629_246_976,
            ModelVariant::Custom => 0,
        }
    }

    /// Return all known (non-Custom) variants.
    pub fn known_variants() -> &'static [ModelVariant] {
        &[
            ModelVariant::Bonsai8B,
            ModelVariant::Bonsai4B,
            ModelVariant::Bonsai1_7B,
            ModelVariant::TernaryBonsai8B,
            ModelVariant::TernaryBonsai4B,
            ModelVariant::TernaryBonsai1_7B,
            ModelVariant::FP8Bonsai8B,
            ModelVariant::FP8Bonsai4B,
            ModelVariant::FP8Bonsai1_7B,
            ModelVariant::Bonsai27B,
            ModelVariant::TernaryBonsai227bPq2,
            ModelVariant::TernaryBonsai227bPtq1,
            ModelVariant::TernaryBonsai227bQ2g64,
            ModelVariant::Bonsai227bMmproj,
        ]
    }

    /// Whether this variant is a known (non-custom) architecture.
    pub fn is_known(&self) -> bool {
        !matches!(self, ModelVariant::Custom)
    }
}

impl std::fmt::Display for ModelVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_bonsai_8b() {
        let config = Qwen3Config::bonsai_8b();
        assert_eq!(ModelVariant::from_config(&config), ModelVariant::Bonsai8B);
        assert_eq!(ModelVariant::Bonsai8B.name(), "Bonsai-8B");
        assert!(ModelVariant::Bonsai8B.is_known());
    }

    #[test]
    fn detect_bonsai_4b() {
        let config = Qwen3Config::bonsai_4b();
        assert_eq!(ModelVariant::from_config(&config), ModelVariant::Bonsai4B);
        assert_eq!(ModelVariant::Bonsai4B.name(), "Bonsai-4B");
        assert!(ModelVariant::Bonsai4B.is_known());
    }

    #[test]
    fn detect_bonsai_1_7b() {
        let config = Qwen3Config::bonsai_1_7b();
        assert_eq!(ModelVariant::from_config(&config), ModelVariant::Bonsai1_7B);
        assert_eq!(ModelVariant::Bonsai1_7B.name(), "Bonsai-1.7B");
        assert!(ModelVariant::Bonsai1_7B.is_known());
    }

    #[test]
    fn detect_custom() {
        let mut config = Qwen3Config::bonsai_8b();
        config.num_layers = 48;
        config.hidden_size = 8192;
        assert_eq!(ModelVariant::from_config(&config), ModelVariant::Custom);
        assert_eq!(ModelVariant::Custom.name(), "Custom");
        assert!(!ModelVariant::Custom.is_known());
    }

    #[test]
    fn default_configs_roundtrip() {
        // Only the 1-bit variants can round-trip through from_config() alone.
        // Ternary variants share the same architecture as their 1-bit siblings,
        // so from_config() returns the 1-bit sibling — that is expected and correct.
        // Ternary detection requires from_config_and_sample_tensor_type().
        let one_bit_variants = [
            ModelVariant::Bonsai8B,
            ModelVariant::Bonsai4B,
            ModelVariant::Bonsai1_7B,
        ];
        for variant in &one_bit_variants {
            let config = variant.default_config();
            let detected = ModelVariant::from_config(&config);
            assert_eq!(
                *variant, detected,
                "variant {:?} config should round-trip",
                variant
            );
        }
    }

    #[test]
    fn param_counts_are_reasonable() {
        assert!(ModelVariant::Bonsai8B.param_count() > 7_000_000_000);
        assert!(ModelVariant::Bonsai8B.param_count() < 10_000_000_000);

        assert!(ModelVariant::Bonsai4B.param_count() > 3_000_000_000);
        assert!(ModelVariant::Bonsai4B.param_count() < 5_000_000_000);

        assert!(ModelVariant::Bonsai1_7B.param_count() > 1_000_000_000);
        assert!(ModelVariant::Bonsai1_7B.param_count() < 2_500_000_000);

        assert_eq!(ModelVariant::Custom.param_count(), 0);
    }

    #[test]
    fn model_sizes_decrease_with_variant() {
        let size_8b = ModelVariant::Bonsai8B.expected_model_size_bytes();
        let size_4b = ModelVariant::Bonsai4B.expected_model_size_bytes();
        let size_1_7b = ModelVariant::Bonsai1_7B.expected_model_size_bytes();

        assert!(size_8b > size_4b, "8B should be larger than 4B");
        assert!(size_4b > size_1_7b, "4B should be larger than 1.7B");
        assert!(size_1_7b > 0, "1.7B should have nonzero size");
    }

    #[test]
    fn display_trait() {
        assert_eq!(format!("{}", ModelVariant::Bonsai8B), "Bonsai-8B");
        assert_eq!(format!("{}", ModelVariant::Custom), "Custom");
    }

    #[test]
    fn known_variants_list() {
        let variants = ModelVariant::known_variants();
        assert_eq!(variants.len(), 14);
        assert!(variants.contains(&ModelVariant::Bonsai8B));
        assert!(variants.contains(&ModelVariant::Bonsai4B));
        assert!(variants.contains(&ModelVariant::Bonsai1_7B));
        assert!(variants.contains(&ModelVariant::TernaryBonsai8B));
        assert!(variants.contains(&ModelVariant::TernaryBonsai4B));
        assert!(variants.contains(&ModelVariant::TernaryBonsai1_7B));
        assert!(variants.contains(&ModelVariant::FP8Bonsai8B));
        assert!(variants.contains(&ModelVariant::FP8Bonsai4B));
        assert!(variants.contains(&ModelVariant::FP8Bonsai1_7B));
        assert!(variants.contains(&ModelVariant::Bonsai27B));
        assert!(variants.contains(&ModelVariant::TernaryBonsai227bPq2));
        assert!(variants.contains(&ModelVariant::TernaryBonsai227bPtq1));
        assert!(variants.contains(&ModelVariant::TernaryBonsai227bQ2g64));
        assert!(variants.contains(&ModelVariant::Bonsai227bMmproj));
    }

    #[test]
    fn bonsai2_27b_family_names_and_sizes() {
        // M-14: these must not fall through to `Custom`'s zeroed metadata.
        for v in [
            ModelVariant::Bonsai27B,
            ModelVariant::TernaryBonsai227bPq2,
            ModelVariant::TernaryBonsai227bPtq1,
            ModelVariant::TernaryBonsai227bQ2g64,
            ModelVariant::Bonsai227bMmproj,
        ] {
            assert!(v.is_known(), "{v:?} must be known, not Custom");
            assert!(v.param_count() > 0, "{v:?} param_count must be nonzero");
            assert!(
                v.expected_model_size_bytes() > 0,
                "{v:?} expected_model_size_bytes must be nonzero"
            );
            assert_ne!(v.name(), "Custom");
        }
        // The three qwen35 language-model variants share the 27B
        // architecture's scalar shape.
        for v in [
            ModelVariant::Bonsai27B,
            ModelVariant::TernaryBonsai227bPq2,
            ModelVariant::TernaryBonsai227bPtq1,
            ModelVariant::TernaryBonsai227bQ2g64,
        ] {
            let cfg = v.default_config();
            assert_eq!(cfg.num_layers, 64, "{v:?}");
            assert_eq!(cfg.hidden_size, 5120, "{v:?}");
            assert_eq!(cfg.vocab_size, 248320, "{v:?}");
            assert_eq!(cfg.architecture, "qwen35", "{v:?}");
        }
    }

    // ── detect_qwen35_27b: M-14 / cli-01 / cli-16 acceptance ────────────────
    //
    // Synthetic-signal coverage for all 6 real 27B files + the mmproj
    // companion, driven by primitives (never GGUF I/O) so it runs
    // unconditionally; `detect_qwen35_27b_on_real_files` below additionally
    // parses the real files when present.

    #[test]
    fn detect_qwen35_27b_ternary_bonsai_2_27b_pq2_0() {
        // Ternary-Bonsai-2-27B-PQ2_0.gguf
        assert_eq!(
            ModelVariant::detect_qwen35_27b(
                "qwen35",
                64,
                oxibonsai_core::GgufTensorType::PQ2_0,
                true
            ),
            Some(ModelVariant::TernaryBonsai227bPq2)
        );
    }

    #[test]
    fn detect_qwen35_27b_ternary_bonsai_2_27b_ptq1_0() {
        // Ternary-Bonsai-2-27B-PTQ1_0.gguf
        assert_eq!(
            ModelVariant::detect_qwen35_27b(
                "qwen35",
                64,
                oxibonsai_core::GgufTensorType::PTQ1_0,
                true
            ),
            Some(ModelVariant::TernaryBonsai227bPtq1)
        );
    }

    #[test]
    fn detect_qwen35_27b_prism_fork_required_q2_0_g64() {
        // Ternary-Bonsai-2-27B-Q2_0-prism-fork-required.gguf: ggml id 42,
        // RESOLVED to the mainline group-64 reading, with a Hadamard fold.
        assert_eq!(
            ModelVariant::detect_qwen35_27b(
                "qwen35",
                64,
                oxibonsai_core::GgufTensorType::Q2_0G64,
                true
            ),
            Some(ModelVariant::TernaryBonsai227bQ2g64)
        );
    }

    #[test]
    fn detect_qwen35_27b_legacy_g128_q2_0_no_hadamard() {
        // Ternary-Bonsai-27B-Q2_0.gguf: ggml id 42, RESOLVED to the
        // PrismML gen-1 group-128 (d-first) reading, no Hadamard fold.
        assert_eq!(
            ModelVariant::detect_qwen35_27b(
                "qwen35",
                64,
                oxibonsai_core::GgufTensorType::Q2_0G128DFirst,
                false
            ),
            Some(ModelVariant::Bonsai27B)
        );
    }

    #[test]
    fn detect_qwen35_27b_legacy_pq2_0_no_hadamard() {
        // Ternary-Bonsai-27B-PQ2_0.gguf: native PQ2_0 id 142, no Hadamard
        // fold — same generation/variant as the g128 file above, different
        // codec, per design §3.9 ("one ModelVariant per generation").
        assert_eq!(
            ModelVariant::detect_qwen35_27b(
                "qwen35",
                64,
                oxibonsai_core::GgufTensorType::PQ2_0,
                false
            ),
            Some(ModelVariant::Bonsai27B)
        );
    }

    #[test]
    fn detect_qwen35_27b_bonsai_27b_q1_0() {
        // Bonsai-27B-Q1_0.gguf
        assert_eq!(
            ModelVariant::detect_qwen35_27b(
                "qwen35",
                64,
                oxibonsai_core::GgufTensorType::Q1_0_g128,
                false
            ),
            Some(ModelVariant::Bonsai27B)
        );
    }

    #[test]
    fn detect_qwen35_27b_mmproj() {
        // Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf: architecture "clip",
        // clip.vision.block_count = 27; the resolved_type/has_hadamard
        // arguments are irrelevant for this architecture.
        assert_eq!(
            ModelVariant::detect_qwen35_27b(
                "clip",
                27,
                oxibonsai_core::GgufTensorType::Q8_0,
                false
            ),
            Some(ModelVariant::Bonsai227bMmproj)
        );
    }

    #[test]
    fn detect_qwen35_27b_never_guesses_an_unseen_combination() {
        // PTQ1_0 without a Hadamard fold and Q2_0G64 without one are not
        // among the six real files — must not be guessed as any variant.
        assert_eq!(
            ModelVariant::detect_qwen35_27b(
                "qwen35",
                64,
                oxibonsai_core::GgufTensorType::PTQ1_0,
                false
            ),
            None
        );
        assert_eq!(
            ModelVariant::detect_qwen35_27b(
                "qwen35",
                64,
                oxibonsai_core::GgufTensorType::Q2_0G64,
                false
            ),
            None
        );
        // Unknown architecture / wrong block_count must not be guessed either.
        assert_eq!(
            ModelVariant::detect_qwen35_27b(
                "qwen3",
                64,
                oxibonsai_core::GgufTensorType::PQ2_0,
                true
            ),
            None
        );
        assert_eq!(
            ModelVariant::detect_qwen35_27b(
                "qwen35",
                36,
                oxibonsai_core::GgufTensorType::PQ2_0,
                true
            ),
            None
        );
        assert_eq!(
            ModelVariant::detect_qwen35_27b(
                "clip",
                12,
                oxibonsai_core::GgufTensorType::Q8_0,
                false
            ),
            None
        );
    }

    // ── `from_config_and_resolved_sample`: the production entry point that
    // wires `detect_qwen35_27b` into the same call `from_config_and_sample_
    // tensor_type` used to make alone (M-14) ────────────────────────────────

    #[test]
    fn from_config_and_resolved_sample_detects_a_27b_family_instead_of_falling_to_custom() {
        // `from_config_and_sample_tensor_type` alone cannot recognise this:
        // `from_config`'s `(num_layers, hidden_size)` table has no `(64,
        // 5120)` arm, so it would fall to `Custom` -- the exact M-14 defect
        // this wrapper closes.
        let config = ModelVariant::TernaryBonsai227bPq2.default_config();
        assert_eq!(config.num_layers, 64);
        assert_eq!(config.hidden_size, 5120);
        assert_eq!(config.architecture, "qwen35");
        let variant = ModelVariant::from_config_and_resolved_sample(
            &config,
            oxibonsai_core::GgufTensorType::PQ2_0,
            true,
        );
        assert_eq!(variant, ModelVariant::TernaryBonsai227bPq2);
        assert_ne!(variant, ModelVariant::Custom);
        assert!(variant.param_count() > 0, "must not zero out a known model");
        assert!(variant.expected_model_size_bytes() > 0);
    }

    #[test]
    fn from_config_and_resolved_sample_falls_through_to_the_legacy_path() {
        // A non-`qwen35` config must behave exactly like
        // `from_config_and_sample_tensor_type` -- `detect_qwen35_27b`
        // returns `None` and the legacy 8B/4B/1.7B-family detection runs
        // unchanged, `has_hadamard` notwithstanding.
        assert_eq!(
            ModelVariant::from_config_and_resolved_sample(
                &Qwen3Config::bonsai_8b(),
                oxibonsai_core::GgufTensorType::TQ2_0_g128,
                true,
            ),
            ModelVariant::TernaryBonsai8B
        );
        assert_eq!(
            ModelVariant::from_config_and_resolved_sample(
                &Qwen3Config::tiny_test(),
                oxibonsai_core::GgufTensorType::Q1_0_g128,
                false,
            ),
            ModelVariant::Custom
        );
    }

    /// Resolve `sample`'s tensor type exactly as a real loader must
    /// (B2-09): ggml wire id 42 is ambiguous (see
    /// `oxibonsai_core::gguf::quant_resolve`'s module doc), and
    /// `sample.tensor_type` alone is only ever the parse-time guess for it
    /// (always `TQ2_0_g128`) -- never the "never from the raw id 42" input
    /// [`ModelVariant::detect_qwen35_27b`]'s own contract demands. Panics
    /// (test-only) rather than silently falling back to the raw type on any
    /// failure, so a real file this cannot resolve fails the test loudly
    /// instead of quietly re-introducing the guess this closes.
    fn resolve_sample_type_for_detection(
        gguf: &oxibonsai_core::gguf::reader::GgufFile<'_>,
        filename: &str,
        sample: &oxibonsai_core::gguf::tensor_info::TensorInfo,
    ) -> oxibonsai_core::GgufTensorType {
        use oxibonsai_core::gguf::quant_resolve::{
            compute_extents, resolve_type_42_with_sample, AMBIGUOUS_TYPE_ID,
        };
        use oxibonsai_core::quant_ternary::{sniff_sample_byte_cap, SNIFF_DEFAULT_BLOCKS};

        if sample.tensor_type.wire_id() != AMBIGUOUS_TYPE_ID {
            return sample.tensor_type;
        }
        let infos: Vec<_> = gguf
            .tensors
            .sorted_by_offset()
            .into_iter()
            .cloned()
            .collect();
        let data_len = (gguf.data.len() as u64).saturating_sub(gguf.data_offset as u64);
        let extents = compute_extents(&infos, data_len)
            .unwrap_or_else(|e| panic!("{filename}: compute_extents failed: {e}"));
        let alignment = gguf
            .metadata
            .get("general.alignment")
            .and_then(|v| v.as_u32())
            .unwrap_or(32) as u64;
        let raw_sample = gguf
            .tensor_data(&sample.name)
            .unwrap_or_else(|e| panic!("{filename}: tensor_data({}) failed: {e}", sample.name));
        let capped = &raw_sample[..raw_sample
            .len()
            .min(sniff_sample_byte_cap(SNIFF_DEFAULT_BLOCKS))];
        let resolved = resolve_type_42_with_sample(
            &infos,
            alignment,
            gguf.metadata.get("general.quantization_version"),
            Some(&extents),
            capped,
        )
        .unwrap_or_else(|e| {
            panic!(
                "{filename}: id-42 resolution of {} failed: {e}",
                sample.name
            )
        });
        resolved.tensor_type
    }

    /// Full end-to-end detection against the six real 27B GGUFs plus the
    /// three legacy (1.7B/8B/4B-family) files, when present on disk.
    /// Model weights are not in the repo (multi-GB), so this skips per file
    /// when absent — the same runtime-skip convention
    /// `gguf_loader.rs::real_legacy_model_resolves_to_the_qs_first_reading`
    /// already uses, rather than a blanket `#[ignore]` that stays skipped
    /// even when the file *is* present.
    ///
    /// Set `OXI_REQUIRE_MODEL_FILES=1` to turn a missing file into a hard
    /// failure instead of a skip (gatekeeper minor finding): a green run
    /// with `models/` containing only `.gitkeep` executes zero assertions,
    /// so CI that actually mounts the real weights should be able to demand
    /// that every case really ran.
    #[test]
    fn detect_qwen35_27b_on_real_files() {
        use oxibonsai_core::gguf::reader::GgufFile;

        let require_real_files = std::env::var("OXI_REQUIRE_MODEL_FILES")
            .map(|v| v == "1")
            .unwrap_or(false);
        let models_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");

        let cases_27b: &[(&str, ModelVariant)] = &[
            (
                "Ternary-Bonsai-2-27B-PQ2_0.gguf",
                ModelVariant::TernaryBonsai227bPq2,
            ),
            (
                "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
                ModelVariant::TernaryBonsai227bPtq1,
            ),
            (
                "Ternary-Bonsai-2-27B-Q2_0-prism-fork-required.gguf",
                ModelVariant::TernaryBonsai227bQ2g64,
            ),
            ("Ternary-Bonsai-27B-Q2_0.gguf", ModelVariant::Bonsai27B),
            ("Ternary-Bonsai-27B-PQ2_0.gguf", ModelVariant::Bonsai27B),
            ("Bonsai-27B-Q1_0.gguf", ModelVariant::Bonsai27B),
        ];
        for (filename, expected) in cases_27b {
            let path = models_dir.join(filename);
            let Ok(bytes) = std::fs::read(&path) else {
                assert!(
                    !require_real_files,
                    "OXI_REQUIRE_MODEL_FILES=1: {filename} must be present at {}",
                    path.display()
                );
                eprintln!("skipping {filename}: not present at {}", path.display());
                continue;
            };
            let gguf = GgufFile::parse(&bytes)
                .unwrap_or_else(|e| panic!("{filename}: real file must parse cleanly: {e}"));
            let arch = gguf
                .metadata
                .get_string("general.architecture")
                .unwrap_or_else(|e| panic!("{filename}: missing general.architecture: {e}"));
            let block_count = gguf
                .metadata
                .get_u32(&format!("{arch}.block_count"))
                .unwrap_or_else(|e| panic!("{filename}: missing {arch}.block_count: {e}"))
                as u64;
            let has_hadamard = gguf.metadata.get("prism.hadamard.version").is_some();
            let sample = gguf
                .tensors
                .require("output.weight")
                .unwrap_or_else(|e| panic!("{filename}: missing output.weight: {e}"));
            // B2-09 fix: never feed `detect_qwen35_27b` the raw ambiguous
            // ggml id 42 -- resolve it first, exactly as a real loader must.
            let resolved_type = resolve_sample_type_for_detection(&gguf, filename, sample);
            let detected =
                ModelVariant::detect_qwen35_27b(arch, block_count, resolved_type, has_hadamard);
            assert_eq!(
                detected,
                Some(*expected),
                "{filename}: expected {expected:?}, detected {detected:?} \
                 (arch={arch}, block_count={block_count}, raw_sample_type={:?}, \
                 resolved_sample_type={resolved_type:?}, has_hadamard={has_hadamard})",
                sample.tensor_type
            );
        }

        let mmproj_path = models_dir.join("Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf");
        if let Ok(bytes) = std::fs::read(&mmproj_path) {
            let gguf = GgufFile::parse(&bytes).expect("mmproj file must parse cleanly");
            let arch = gguf
                .metadata
                .get_string("general.architecture")
                .expect("mmproj missing general.architecture");
            let block_count = gguf
                .metadata
                .get_u32(&format!("{arch}.vision.block_count"))
                .or_else(|_| gguf.metadata.get_u32(&format!("{arch}.block_count")))
                .expect("mmproj missing block_count");
            let detected = ModelVariant::detect_qwen35_27b(
                arch,
                block_count as u64,
                oxibonsai_core::GgufTensorType::Q8_0,
                false,
            );
            assert_eq!(detected, Some(ModelVariant::Bonsai227bMmproj));
        } else {
            assert!(
                !require_real_files,
                "OXI_REQUIRE_MODEL_FILES=1: mmproj must be present at {}",
                mmproj_path.display()
            );
            eprintln!("skipping mmproj: not present at {}", mmproj_path.display());
        }
    }

    /// Regression (design §7.4): the three legacy (non-27B) local models
    /// must keep resolving exactly as before this package's changes. Model
    /// weights are not in the repo, so this skips per file when absent.
    ///
    /// Set `OXI_REQUIRE_MODEL_FILES=1` to turn a missing file into a hard
    /// failure instead of a skip (same convention as
    /// `detect_qwen35_27b_on_real_files`).
    #[test]
    fn legacy_model_variants_unaffected_by_27b_detection() {
        use oxibonsai_core::gguf::reader::GgufFile;

        let require_real_files = std::env::var("OXI_REQUIRE_MODEL_FILES")
            .map(|v| v == "1")
            .unwrap_or(false);
        let models_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
        let cases: &[(&str, ModelVariant)] = &[
            ("Ternary-Bonsai-1.7B.gguf", ModelVariant::TernaryBonsai1_7B),
            ("Ternary-Bonsai-8B.gguf", ModelVariant::TernaryBonsai8B),
            ("Bonsai-8B.gguf", ModelVariant::Bonsai8B),
        ];
        for (filename, expected) in cases {
            let path = models_dir.join(filename);
            let Ok(bytes) = std::fs::read(&path) else {
                assert!(
                    !require_real_files,
                    "OXI_REQUIRE_MODEL_FILES=1: {filename} must be present at {}",
                    path.display()
                );
                eprintln!("skipping {filename}: not present at {}", path.display());
                continue;
            };
            let gguf = GgufFile::parse(&bytes)
                .unwrap_or_else(|e| panic!("{filename}: real file must parse cleanly: {e}"));
            let config = Qwen3Config::from_metadata(&gguf.metadata)
                .unwrap_or_else(|e| panic!("{filename}: config parse failed: {e}"));
            let sample = gguf
                .tensors
                .require("output.weight")
                .unwrap_or_else(|e| panic!("{filename}: missing output.weight: {e}"));
            let detected =
                ModelVariant::from_config_and_sample_tensor_type(&config, sample.tensor_type);
            assert_eq!(detected, *expected, "{filename}");
            assert_ne!(
                detected,
                ModelVariant::Custom,
                "{filename}: must not fall to Custom with zeroed metadata"
            );
        }
    }

    #[test]
    fn detect_ternary_8b_by_tensor_type() {
        let cfg = Qwen3Config::ternary_bonsai_8b();
        let variant = ModelVariant::from_config_and_sample_tensor_type(
            &cfg,
            oxibonsai_core::GgufTensorType::TQ2_0_g128,
        );
        assert_eq!(variant, ModelVariant::TernaryBonsai8B);
    }

    #[test]
    fn detect_bonsai_8b_stays_1bit() {
        let cfg = Qwen3Config::bonsai_8b();
        let variant = ModelVariant::from_config_and_sample_tensor_type(
            &cfg,
            oxibonsai_core::GgufTensorType::Q1_0_g128,
        );
        assert_eq!(variant, ModelVariant::Bonsai8B);
    }

    #[test]
    fn ternary_variant_param_counts_match_bonsai() {
        assert_eq!(
            ModelVariant::TernaryBonsai8B.param_count(),
            ModelVariant::Bonsai8B.param_count()
        );
        assert_eq!(
            ModelVariant::TernaryBonsai4B.param_count(),
            ModelVariant::Bonsai4B.param_count()
        );
        assert_eq!(
            ModelVariant::TernaryBonsai1_7B.param_count(),
            ModelVariant::Bonsai1_7B.param_count()
        );
    }

    #[test]
    fn ternary_variant_expected_size_less_than_fp16() {
        // Ternary 8B at ~1.75 GB should be way less than FP16 8B at ~16 GB
        let ternary_size = ModelVariant::TernaryBonsai8B.expected_model_size_bytes();
        assert!(
            ternary_size < 2_000_000_000,
            "8B ternary expected < 2 GB, got {}",
            ternary_size
        );
        assert!(
            ternary_size > 1_000_000_000,
            "8B ternary expected > 1 GB, got {}",
            ternary_size
        );
    }

    #[test]
    fn ternary_variants_are_known() {
        assert!(ModelVariant::TernaryBonsai8B.is_known());
        assert!(ModelVariant::TernaryBonsai4B.is_known());
        assert!(ModelVariant::TernaryBonsai1_7B.is_known());
    }

    #[test]
    fn ternary_variant_names() {
        assert_eq!(ModelVariant::TernaryBonsai8B.name(), "Ternary-Bonsai-8B");
        assert_eq!(ModelVariant::TernaryBonsai4B.name(), "Ternary-Bonsai-4B");
        assert_eq!(
            ModelVariant::TernaryBonsai1_7B.name(),
            "Ternary-Bonsai-1.7B"
        );
    }

    #[test]
    fn ternary_display_trait() {
        assert_eq!(
            format!("{}", ModelVariant::TernaryBonsai8B),
            "Ternary-Bonsai-8B"
        );
        assert_eq!(
            format!("{}", ModelVariant::TernaryBonsai4B),
            "Ternary-Bonsai-4B"
        );
        assert_eq!(
            format!("{}", ModelVariant::TernaryBonsai1_7B),
            "Ternary-Bonsai-1.7B"
        );
    }

    #[test]
    fn ternary_default_configs_roundtrip() {
        // Ternary variants have identical architecture to their 1-bit siblings,
        // so from_config() returns the 1-bit variant — that is expected and correct.
        // Verify the default_config() returns sensible configs with matching architecture.
        let cfg_8b = ModelVariant::TernaryBonsai8B.default_config();
        assert_eq!(cfg_8b.num_layers, 36);
        assert_eq!(cfg_8b.hidden_size, 4096);

        let cfg_4b = ModelVariant::TernaryBonsai4B.default_config();
        assert_eq!(cfg_4b.num_layers, 24);
        assert_eq!(cfg_4b.hidden_size, 2560);

        // Real GGUF header values (models/Ternary-Bonsai-1.7B.gguf):
        // 28 layers, hidden_size 2048 (see Qwen3Config::bonsai_1_7b()).
        let cfg_1_7b = ModelVariant::TernaryBonsai1_7B.default_config();
        assert_eq!(cfg_1_7b.num_layers, 28);
        assert_eq!(cfg_1_7b.hidden_size, 2048);
    }

    #[test]
    fn detect_ternary_4b_and_1_7b_by_tensor_type() {
        let cfg_4b = Qwen3Config::ternary_bonsai_4b();
        let variant_4b = ModelVariant::from_config_and_sample_tensor_type(
            &cfg_4b,
            oxibonsai_core::GgufTensorType::TQ2_0_g128,
        );
        assert_eq!(variant_4b, ModelVariant::TernaryBonsai4B);

        let cfg_1_7b = Qwen3Config::ternary_bonsai_1_7b();
        let variant_1_7b = ModelVariant::from_config_and_sample_tensor_type(
            &cfg_1_7b,
            oxibonsai_core::GgufTensorType::TQ2_0_g128,
        );
        assert_eq!(variant_1_7b, ModelVariant::TernaryBonsai1_7B);
    }

    #[test]
    fn custom_stays_custom_with_ternary_type() {
        let mut cfg = Qwen3Config::bonsai_8b();
        cfg.num_layers = 48;
        cfg.hidden_size = 8192;
        let variant = ModelVariant::from_config_and_sample_tensor_type(
            &cfg,
            oxibonsai_core::GgufTensorType::TQ2_0_g128,
        );
        assert_eq!(variant, ModelVariant::Custom);
    }

    #[test]
    fn detect_fp8_e4m3_8b_by_tensor_type() {
        let cfg = Qwen3Config::bonsai_8b();
        let variant = ModelVariant::from_config_and_sample_tensor_type(
            &cfg,
            oxibonsai_core::GgufTensorType::F8_E4M3,
        );
        assert_eq!(variant, ModelVariant::FP8Bonsai8B);
    }

    #[test]
    fn detect_fp8_e5m2_1_7b_by_tensor_type() {
        let cfg = Qwen3Config::bonsai_1_7b();
        let variant = ModelVariant::from_config_and_sample_tensor_type(
            &cfg,
            oxibonsai_core::GgufTensorType::F8_E5M2,
        );
        assert_eq!(variant, ModelVariant::FP8Bonsai1_7B);
    }

    #[test]
    fn fp8_variant_param_counts_match_bonsai() {
        assert_eq!(
            ModelVariant::FP8Bonsai8B.param_count(),
            ModelVariant::Bonsai8B.param_count()
        );
        assert_eq!(
            ModelVariant::FP8Bonsai4B.param_count(),
            ModelVariant::Bonsai4B.param_count()
        );
        assert_eq!(
            ModelVariant::FP8Bonsai1_7B.param_count(),
            ModelVariant::Bonsai1_7B.param_count()
        );
    }

    #[test]
    fn fp8_variant_names() {
        assert_eq!(ModelVariant::FP8Bonsai8B.name(), "FP8-Bonsai-8B");
        assert_eq!(ModelVariant::FP8Bonsai4B.name(), "FP8-Bonsai-4B");
        assert_eq!(ModelVariant::FP8Bonsai1_7B.name(), "FP8-Bonsai-1.7B");
    }

    #[test]
    fn fp8_variants_are_known() {
        assert!(ModelVariant::FP8Bonsai8B.is_known());
        assert!(ModelVariant::FP8Bonsai4B.is_known());
        assert!(ModelVariant::FP8Bonsai1_7B.is_known());
    }
}
