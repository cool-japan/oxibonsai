//! Builder pattern for constructing [`Qwen3Config`] values.
//!
//! [`ModelConfigBuilder`] provides a fluent, ergonomic API for assembling
//! custom model configurations without needing to fill every field manually.
//! All fields default to the Bonsai-8B values when not explicitly set so
//! that partial configurations remain valid.
//!
//! # Examples
//!
//! ```rust
//! use oxibonsai_model::model_config_builder::ModelConfigBuilder;
//!
//! // Build a custom tiny config for unit tests
//! let config = ModelConfigBuilder::build_tiny();
//! assert_eq!(config.num_layers, 2);
//!
//! // Build a custom config with the builder API
//! let config = ModelConfigBuilder::new()
//!     .layers(8)
//!     .hidden_size(512)
//!     .num_attention_heads(8)
//!     .num_kv_heads(2)
//!     .intermediate_size(1024)
//!     .vocab_size(1000)
//!     .max_position_embeddings(2048)
//!     .rope_freq_base(10_000.0)
//!     .rms_norm_eps(1e-6)
//!     .build()
//!     .expect("valid config");
//!
//! assert_eq!(config.num_layers, 8);
//! assert_eq!(config.hidden_size, 512);
//! ```

use oxibonsai_core::config::{Qwen3Config, RopeScaling};

// ─── ConfigError ─────────────────────────────────────────────────────────────

/// Error produced when a [`ModelConfigBuilder`] constraint is violated.
///
/// Carries a human-readable description of the violated constraint so callers
/// can surface actionable diagnostics without needing pattern-matching.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError(pub String);

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid model configuration: {}", self.0)
    }
}

impl std::error::Error for ConfigError {}

impl ConfigError {
    /// Construct a new error with the provided message.
    fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

// ─── ModelConfigBuilder ───────────────────────────────────────────────────────

/// Fluent builder for [`Qwen3Config`].
///
/// Each setter consumes the builder by value to enable method-chaining without
/// requiring mutable references. Call [`build`][ModelConfigBuilder::build] to
/// validate the accumulated settings and produce a [`Qwen3Config`].
///
/// Fields left unset are filled from the Bonsai-8B defaults.
#[derive(Debug, Default, Clone)]
pub struct ModelConfigBuilder {
    num_layers: Option<usize>,
    hidden_size: Option<usize>,
    num_attention_heads: Option<usize>,
    num_kv_heads: Option<usize>,
    intermediate_size: Option<usize>,
    vocab_size: Option<usize>,
    max_position_embeddings: Option<usize>,
    rope_freq_base: Option<f32>,
    rms_norm_eps: Option<f32>,
    architecture: Option<String>,
    model_name: Option<String>,
    head_dim: Option<usize>,
    sliding_window: Option<usize>,
}

impl ModelConfigBuilder {
    /// Create a new builder with all fields unset (will use Bonsai-8B defaults).
    pub fn new() -> Self {
        Self::default()
    }

    // ── Setters ──────────────────────────────────────────────────────────────

    /// Set the number of Transformer layers.
    pub fn layers(mut self, n: usize) -> Self {
        self.num_layers = Some(n);
        self
    }

    /// Set the hidden (embedding) dimension.
    pub fn hidden_size(mut self, n: usize) -> Self {
        self.hidden_size = Some(n);
        self
    }

    /// Set the number of query attention heads.
    pub fn num_attention_heads(mut self, n: usize) -> Self {
        self.num_attention_heads = Some(n);
        self
    }

    /// Set the number of key-value heads (for Grouped Query Attention).
    pub fn num_kv_heads(mut self, n: usize) -> Self {
        self.num_kv_heads = Some(n);
        self
    }

    /// Set the intermediate (FFN / SwiGLU) size.
    pub fn intermediate_size(mut self, n: usize) -> Self {
        self.intermediate_size = Some(n);
        self
    }

    /// Set the vocabulary size.
    pub fn vocab_size(mut self, n: usize) -> Self {
        self.vocab_size = Some(n);
        self
    }

    /// Set the maximum position embedding length (= maximum context length).
    pub fn max_position_embeddings(mut self, n: usize) -> Self {
        self.max_position_embeddings = Some(n);
        self
    }

    /// Set the RoPE frequency base (theta).
    pub fn rope_freq_base(mut self, f: f32) -> Self {
        self.rope_freq_base = Some(f);
        self
    }

    /// Set the RMSNorm epsilon.
    pub fn rms_norm_eps(mut self, f: f32) -> Self {
        self.rms_norm_eps = Some(f);
        self
    }

    /// Override the architecture tag stored in the resulting config.
    pub fn architecture(mut self, s: impl Into<String>) -> Self {
        self.architecture = Some(s.into());
        self
    }

    /// Override the model name stored in the resulting config.
    pub fn model_name(mut self, s: impl Into<String>) -> Self {
        self.model_name = Some(s.into());
        self
    }

    /// Explicitly set `head_dim`, independent of `hidden_size /
    /// num_attention_heads` (M-24).
    ///
    /// Most architectures derive `head_dim` from `hidden_size /
    /// num_attention_heads`, and that remains the default when this setter
    /// is not called. Some architectures cannot be expressed that way —
    /// e.g. Bonsai 2 (27B): `hidden_size = 5120`, `num_attention_heads =
    /// 24`, `head_dim = 256`, and `5120 / 24` is not even an integer. Call
    /// this to override the derived value; when set, [`Self::build`] also
    /// skips its "hidden_size must be divisible by num_attention_heads"
    /// check, since that check exists solely to keep the *derived* value
    /// sane and no longer applies once `head_dim` is explicit.
    pub fn head_dim(mut self, n: usize) -> Self {
        self.head_dim = Some(n);
        self
    }

    /// Restrict attention to the `n` most recent key positions (M-17).
    ///
    /// The builder's counterpart to the GGUF key
    /// `<arch>.attention.sliding_window`: a model built through this builder
    /// can now express local attention exactly as one loaded from a file
    /// can, instead of silently being fully causal. Leaving it unset (the
    /// default) means fully-causal attention over the whole context, which
    /// is what every shipped model declares.
    ///
    /// `n == 0` is refused by [`Self::build`] (via
    /// [`Qwen3Config::validate`]), not folded into "no window": a zero-width
    /// window admits no key positions at all.
    pub fn sliding_window(mut self, n: usize) -> Self {
        self.sliding_window = Some(n);
        self
    }

    // ── Build ─────────────────────────────────────────────────────────────────

    /// Validate the accumulated settings and produce a [`Qwen3Config`].
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] when any of the following constraints are
    /// violated. All but the first two now live once in
    /// [`Qwen3Config::validate`] (oxibonsai-core, M-24/M-12 consolidation)
    /// and are shared with [`Qwen3Config::from_metadata`] -- this builder
    /// converts that shared [`BonsaiError`](oxibonsai_core::error::BonsaiError)
    /// into a [`ConfigError`] via [`ToString`].
    ///
    /// | Constraint | Reason |
    /// |------------|--------|
    /// | `num_attention_heads >= 1` | checked HERE, before this builder's own `hidden_size / num_attention_heads` division |
    /// | `hidden_size` divisible by `num_attention_heads`, *unless* `.head_dim(n)` was called (M-24) | checked HERE: only meaningful on the unmerged `Option`, which `validate()` never sees |
    /// | `num_layers >= 1` | A transformer with zero layers is meaningless |
    /// | `hidden_size >= 1` | Zero-dimensional embeddings are invalid |
    /// | `num_kv_heads >= 1` | At least one KV head is required |
    /// | `head_dim >= 1` | Zero-width heads are invalid |
    /// | `num_attention_heads` divisible by `num_kv_heads` | GQA requirement |
    /// | `intermediate_size >= 1` | FFN must have positive width |
    /// | `vocab_size >= 2` | At least 2 tokens needed for meaningful output |
    /// | `sliding_window != Some(0)` (M-17) | A zero-width attention window admits no key positions |
    /// | `max_context_length >= 1` (this builder's `.max_position_embeddings(n)`) | Context must be at least 1 token |
    /// | `rope_freq_base > 0` | Must be a positive real number |
    /// | `rms_norm_eps > 0` | Epsilon must be strictly positive |
    pub fn build(self) -> Result<Qwen3Config, ConfigError> {
        // Merge with defaults
        let defaults = Qwen3Config::bonsai_8b();

        let num_layers = self.num_layers.unwrap_or(defaults.num_layers);
        let hidden_size = self.hidden_size.unwrap_or(defaults.hidden_size);
        let num_attention_heads = self
            .num_attention_heads
            .unwrap_or(defaults.num_attention_heads);
        let num_kv_heads = self.num_kv_heads.unwrap_or(defaults.num_kv_heads);
        let intermediate_size = self.intermediate_size.unwrap_or(defaults.intermediate_size);
        let vocab_size = self.vocab_size.unwrap_or(defaults.vocab_size);
        let max_context_length = self
            .max_position_embeddings
            .unwrap_or(defaults.max_context_length);
        let rope_freq_base = self.rope_freq_base.unwrap_or(defaults.rope_freq_base);
        let rms_norm_eps = self.rms_norm_eps.unwrap_or(defaults.rms_norm_eps);
        let architecture = self
            .architecture
            .unwrap_or_else(|| defaults.architecture.clone());
        let model_name = self
            .model_name
            .unwrap_or_else(|| defaults.model_name.clone());

        // ── Validation ────────────────────────────────────────────────────────
        //
        // M-24/M-12 consolidation: every check that can run on a fully
        // resolved `Qwen3Config` now lives once in `Qwen3Config::validate`
        // (oxibonsai-core) and is shared with `Qwen3Config::from_metadata`.
        // Exactly one check stays HERE, ahead of that call: "hidden_size
        // divisible by num_attention_heads" only makes sense while
        // `head_dim` is still the unmerged `Option` this builder is holding
        // -- once `head_dim` is resolved into a plain `usize` below, whether
        // it came from `.head_dim(n)` or from the division is no longer
        // recoverable, so `validate()` itself deliberately does not (and
        // cannot) re-derive this constraint.
        //
        // `num_attention_heads == 0` is *also* still checked here (ahead of
        // `validate()`, which is unreachable if this panics first): the
        // fallback division below (`hidden_size / num_attention_heads`)
        // would divide by zero before `validate()` ever ran otherwise.
        if num_attention_heads == 0 {
            return Err(ConfigError::new("num_attention_heads must be >= 1"));
        }
        // M-24: this check only exists to keep the *derived*
        // `hidden_size / num_attention_heads` head_dim sane. When
        // `.head_dim(n)` was called explicitly, the derived value is never
        // used, so an architecture like Bonsai 2 (hidden_size=5120,
        // num_attention_heads=24, head_dim=256; 5120 % 24 != 0) must not be
        // rejected here.
        if self.head_dim.is_none() && !hidden_size.is_multiple_of(num_attention_heads) {
            return Err(ConfigError::new(format!(
                "hidden_size ({hidden_size}) must be divisible by num_attention_heads \
                 ({num_attention_heads}) when head_dim is not set explicitly \
                 (call `.head_dim(n)` for architectures where head_dim != \
                 hidden_size / num_attention_heads)"
            )));
        }

        // M-24: an explicit `.head_dim(n)` always wins over the derived
        // value — see the setter's doc comment.
        let head_dim = self.head_dim.unwrap_or(hidden_size / num_attention_heads);

        let config = Qwen3Config {
            hidden_size,
            intermediate_size,
            num_layers,
            num_attention_heads,
            num_kv_heads,
            head_dim,
            value_length: head_dim,
            vocab_size,
            max_context_length,
            rms_norm_eps,
            rope_freq_base,
            rope_scaling: RopeScaling::None,
            sliding_window: self.sliding_window,
            architecture,
            model_name,
        };
        config
            .validate()
            .map_err(|e| ConfigError::new(e.to_string()))?;
        Ok(config)
    }

    // ── Convenience constructors ──────────────────────────────────────────────

    /// Produce a minimal valid [`Qwen3Config`] suitable for fast unit tests.
    ///
    /// Uses the same parameters as [`Qwen3Config::tiny_test`] so that tests
    /// written against this builder's output are directly comparable to tests
    /// using the core crate's constant.
    pub fn build_tiny() -> Qwen3Config {
        // Uses exact same values as Qwen3Config::tiny_test() for consistency.
        // We bypass the builder to avoid any possibility of validation failure
        // in a shared test helper.
        Qwen3Config::tiny_test()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Happy-path: valid builds ──────────────────────────────────────────────

    #[test]
    fn build_with_all_defaults_succeeds() {
        let config = ModelConfigBuilder::new()
            .build()
            .expect("default builder should succeed");
        // Defaults match Bonsai-8B
        assert_eq!(config.num_layers, 36);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_kv_heads, 8);
        // Real GGUF header value (models/*Bonsai-8B.gguf): vocab 151669.
        assert_eq!(config.vocab_size, 151_669);
    }

    #[test]
    fn build_custom_small_config_succeeds() {
        let config = ModelConfigBuilder::new()
            .layers(4)
            .hidden_size(256)
            .num_attention_heads(4)
            .num_kv_heads(2)
            .intermediate_size(512)
            .vocab_size(100)
            .max_position_embeddings(512)
            .rope_freq_base(10_000.0)
            .rms_norm_eps(1e-5)
            .build()
            .expect("small valid config should build");

        assert_eq!(config.num_layers, 4);
        assert_eq!(config.hidden_size, 256);
        assert_eq!(config.num_attention_heads, 4);
        assert_eq!(config.num_kv_heads, 2);
        assert_eq!(config.intermediate_size, 512);
        assert_eq!(config.vocab_size, 100);
        assert_eq!(config.max_context_length, 512);
        assert!((config.rope_freq_base - 10_000.0).abs() < 1.0);
        assert!((config.rms_norm_eps - 1e-5).abs() < 1e-10);
        // Derived field
        assert_eq!(config.head_dim, 64); // 256 / 4
    }

    #[test]
    fn build_tiny_returns_valid_config() {
        let config = ModelConfigBuilder::build_tiny();
        assert_eq!(config.num_layers, 2);
        assert_eq!(config.hidden_size, 64);
        assert_eq!(config.num_attention_heads, 4);
        assert_eq!(config.num_kv_heads, 2);
        // head_dim derived correctly
        assert_eq!(config.head_dim, 16);
    }

    #[test]
    fn architecture_and_model_name_setters_work() {
        let config = ModelConfigBuilder::new()
            .layers(2)
            .hidden_size(64)
            .num_attention_heads(4)
            .num_kv_heads(2)
            .intermediate_size(128)
            .vocab_size(1000)
            .architecture("custom_arch")
            .model_name("My-Model")
            .build()
            .expect("should build");
        assert_eq!(config.architecture, "custom_arch");
        assert_eq!(config.model_name, "My-Model");
    }

    // ── M-24: explicit head_dim ────────────────────────────────────────────────

    #[test]
    fn explicit_head_dim_overrides_derived_value() {
        // hidden_size=256, heads=4 would derive head_dim=64; explicit
        // head_dim=100 must win instead.
        let config = ModelConfigBuilder::new()
            .layers(2)
            .hidden_size(256)
            .num_attention_heads(4)
            .num_kv_heads(2)
            .intermediate_size(512)
            .vocab_size(100)
            .head_dim(100)
            .build()
            .expect("explicit head_dim should build");
        assert_eq!(config.head_dim, 100);
    }

    // ── M-17: explicit sliding window ──────────────────────────────────────────

    #[test]
    fn sliding_window_defaults_to_none() {
        let config = ModelConfigBuilder::new()
            .build()
            .expect("defaults should build");
        assert_eq!(config.sliding_window, None);
    }

    #[test]
    fn explicit_sliding_window_is_carried_through() {
        let config = ModelConfigBuilder::new()
            .sliding_window(4096)
            .build()
            .expect("explicit sliding_window should build");
        assert_eq!(config.sliding_window, Some(4096));
    }

    #[test]
    fn zero_sliding_window_is_rejected() {
        let err = ModelConfigBuilder::new()
            .sliding_window(0)
            .build()
            .expect_err("a zero-width window must not build");
        assert!(
            err.to_string().contains("sliding_window"),
            "error should name the offending field: {err}"
        );
    }

    #[test]
    fn explicit_head_dim_allows_non_divisible_hidden_size() {
        // The real Bonsai 2 (27B) shape: hidden_size=5120,
        // num_attention_heads=24 (5120 % 24 != 0), head_dim=256,
        // num_kv_heads=4 (24 % 4 == 0, GQA-valid).
        let config = ModelConfigBuilder::new()
            .layers(64)
            .hidden_size(5120)
            .num_attention_heads(24)
            .num_kv_heads(4)
            .intermediate_size(17408)
            .vocab_size(248320)
            .head_dim(256)
            .build()
            .expect("Bonsai 2 shape should build with explicit head_dim");
        assert_eq!(config.head_dim, 256);
        assert_eq!(config.hidden_size, 5120);
        assert_eq!(config.num_attention_heads, 24);
    }

    #[test]
    fn without_explicit_head_dim_non_divisible_hidden_size_still_errors() {
        // Unchanged pre-existing behaviour: the derived-head_dim guard still
        // fires when `.head_dim()` was never called.
        let err = ModelConfigBuilder::new()
            .hidden_size(5120)
            .num_attention_heads(24)
            .num_kv_heads(4)
            .build()
            .expect_err("non-divisible hidden/heads without explicit head_dim should fail");
        assert!(err.0.contains("divisible"), "{err}");
    }

    #[test]
    fn explicit_head_dim_zero_returns_error() {
        let err = ModelConfigBuilder::new()
            .layers(2)
            .hidden_size(64)
            .num_attention_heads(4)
            .num_kv_heads(2)
            .intermediate_size(128)
            .vocab_size(100)
            .head_dim(0)
            .build()
            .expect_err("head_dim=0 should fail");
        assert!(err.0.contains("head_dim"), "{err}");
    }

    #[test]
    fn default_head_dim_still_derived_when_not_set() {
        let config = ModelConfigBuilder::new()
            .layers(2)
            .hidden_size(256)
            .num_attention_heads(4)
            .num_kv_heads(2)
            .intermediate_size(512)
            .vocab_size(100)
            .build()
            .expect("default derivation should still work");
        assert_eq!(config.head_dim, 64); // 256 / 4, unchanged pre-M-24 behaviour
    }

    #[test]
    fn partial_override_inherits_defaults() {
        // Only override layers; everything else should come from Bonsai-8B defaults
        let config = ModelConfigBuilder::new()
            .layers(12)
            .build()
            .expect("partial override should succeed");
        assert_eq!(config.num_layers, 12);
        assert_eq!(config.hidden_size, 4096); // default
                                              // Real GGUF header value (models/*Bonsai-8B.gguf): vocab 151669.
        assert_eq!(config.vocab_size, 151_669); // default
    }

    // ── Error cases: invalid builds ───────────────────────────────────────────

    #[test]
    fn zero_layers_returns_error() {
        let err = ModelConfigBuilder::new()
            .layers(0)
            .build()
            .expect_err("zero layers should fail");
        assert!(
            err.0.contains("num_layers"),
            "error should mention field: {err}"
        );
    }

    #[test]
    fn zero_hidden_size_returns_error() {
        let err = ModelConfigBuilder::new()
            .hidden_size(0)
            .build()
            .expect_err("zero hidden_size should fail");
        assert!(err.0.contains("hidden_size"), "{err}");
    }

    #[test]
    fn zero_attention_heads_returns_error() {
        let err = ModelConfigBuilder::new()
            .num_attention_heads(0)
            .build()
            .expect_err("zero attention heads should fail");
        assert!(err.0.contains("num_attention_heads"), "{err}");
    }

    #[test]
    fn zero_kv_heads_returns_error() {
        let err = ModelConfigBuilder::new()
            .num_kv_heads(0)
            .build()
            .expect_err("zero kv_heads should fail");
        assert!(err.0.contains("num_kv_heads"), "{err}");
    }

    #[test]
    fn hidden_size_not_divisible_by_heads_returns_error() {
        // hidden=100, heads=3 → 100 % 3 ≠ 0
        let err = ModelConfigBuilder::new()
            .hidden_size(100)
            .num_attention_heads(3)
            .num_kv_heads(1)
            .build()
            .expect_err("indivisible hidden/heads should fail");
        assert!(
            err.0.contains("divisible"),
            "error should mention divisibility: {err}"
        );
    }

    #[test]
    fn attention_heads_not_divisible_by_kv_heads_returns_error() {
        // heads=6, kv_heads=4 → 6 % 4 ≠ 0
        let err = ModelConfigBuilder::new()
            .hidden_size(96) // 96 / 6 = 16 (valid head_dim)
            .num_attention_heads(6)
            .num_kv_heads(4)
            .build()
            .expect_err("GQA divisibility violation should fail");
        assert!(
            err.0.contains("divisible"),
            "error should mention divisibility: {err}"
        );
    }

    #[test]
    fn zero_intermediate_size_returns_error() {
        let err = ModelConfigBuilder::new()
            .intermediate_size(0)
            .build()
            .expect_err("zero intermediate_size should fail");
        assert!(err.0.contains("intermediate_size"), "{err}");
    }

    #[test]
    fn vocab_size_one_returns_error() {
        let err = ModelConfigBuilder::new()
            .vocab_size(1)
            .build()
            .expect_err("vocab_size=1 should fail");
        assert!(err.0.contains("vocab_size"), "{err}");
    }

    #[test]
    fn zero_max_position_embeddings_returns_error() {
        let err = ModelConfigBuilder::new()
            .max_position_embeddings(0)
            .build()
            .expect_err("zero max_position_embeddings should fail");
        // M-24/M-12: this check now runs inside the shared
        // `Qwen3Config::validate` (oxibonsai-core), which names the field by
        // the `Qwen3Config` struct's own name (`max_context_length`) rather
        // than this builder's setter name (`max_position_embeddings`) --
        // both describe the same constraint on the same resolved value.
        assert!(err.0.contains("max_context_length"), "{err}");
    }

    #[test]
    fn non_positive_rope_freq_base_returns_error() {
        for bad in [-1.0f32, 0.0, f32::NEG_INFINITY, f32::NAN] {
            let err = ModelConfigBuilder::new()
                .rope_freq_base(bad)
                .build()
                .expect_err(&format!("rope_freq_base={bad} should fail"));
            assert!(err.0.contains("rope_freq_base"), "{err}");
        }
    }

    #[test]
    fn non_positive_rms_norm_eps_returns_error() {
        for bad in [-1e-6f32, 0.0, f32::NEG_INFINITY, f32::NAN] {
            let err = ModelConfigBuilder::new()
                .rms_norm_eps(bad)
                .build()
                .expect_err(&format!("rms_norm_eps={bad} should fail"));
            assert!(err.0.contains("rms_norm_eps"), "{err}");
        }
    }

    // ── ConfigError trait impls ───────────────────────────────────────────────

    #[test]
    fn config_error_display_contains_message() {
        let e = ConfigError::new("test message");
        let s = format!("{e}");
        assert!(s.contains("test message"), "Display should include message");
        assert!(
            s.contains("invalid model configuration"),
            "Display should include prefix"
        );
    }

    #[test]
    fn config_error_is_std_error() {
        let e = ConfigError::new("oops");
        // Verifies that ConfigError implements std::error::Error
        let _: &dyn std::error::Error = &e;
    }
}
