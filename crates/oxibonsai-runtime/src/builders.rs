//! Builder patterns for ergonomic OxiBonsai setup.
//!
//! Three builders are provided for validating and constructing the main
//! runtime objects:
//!
//! - [`SamplerBuilder`] — validates and creates a [`Sampler`]
//! - [`ConfigBuilder`] — validates and creates an [`OxiBonsaiConfig`]
//! - [`EngineBuilder`] — orchestrates config + sampler together

use crate::config::{OxiBonsaiConfig, RopeScalingMode};
use crate::error::{RuntimeError, RuntimeResult};
use crate::sampling::{Sampler, SamplingParams};

/// Builder for sampling parameters with validation.
///
/// # Example
///
/// ```
/// use oxibonsai_runtime::builders::SamplerBuilder;
///
/// let sampler = SamplerBuilder::new()
///     .temperature(0.5)
///     .top_k(50)
///     .top_p(0.95)
///     .repetition_penalty(1.2)
///     .seed(123)
///     .build()
///     .expect("valid parameters");
///
/// let params = sampler.params();
/// assert!((params.temperature - 0.5).abs() < f32::EPSILON);
/// assert_eq!(params.top_k, 50);
/// ```
pub struct SamplerBuilder {
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
    seed: u64,
}

impl SamplerBuilder {
    /// Create a new sampler builder with default values.
    ///
    /// `repetition_penalty` defaults to `1.0` (no-op); it used to be `1.1`,
    /// one of the residual seeds left over after
    /// `sampling::SamplingParams::default()` was corrected to `1.0` (RT-24) —
    /// the P0
    /// CPU-vs-Metal greedy-parity fix depends on every default-constructed
    /// sampler applying no penalty unless a caller opts in explicitly.
    pub fn new() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: 1.0,
            seed: 42,
        }
    }

    /// Set the temperature for softmax scaling. Must be >= 0.
    pub fn temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    /// Set top-k filtering. 0 = disabled.
    pub fn top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// Set top-p (nucleus) threshold. Must be in [0.0, 1.0].
    pub fn top_p(mut self, p: f32) -> Self {
        self.top_p = p;
        self
    }

    /// Set repetition penalty. Must be >= 1.0.
    pub fn repetition_penalty(mut self, rp: f32) -> Self {
        self.repetition_penalty = rp;
        self
    }

    /// Set the random seed.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Validate parameters and build the [`Sampler`].
    pub fn build(self) -> RuntimeResult<Sampler> {
        if self.temperature < 0.0 {
            return Err(RuntimeError::Config(format!(
                "temperature must be >= 0.0, got {}",
                self.temperature
            )));
        }
        if self.top_p < 0.0 || self.top_p > 1.0 {
            return Err(RuntimeError::Config(format!(
                "top_p must be in [0.0, 1.0], got {}",
                self.top_p
            )));
        }
        if self.repetition_penalty < 1.0 {
            return Err(RuntimeError::Config(format!(
                "repetition_penalty must be >= 1.0, got {}",
                self.repetition_penalty
            )));
        }

        let params = SamplingParams {
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            repetition_penalty: self.repetition_penalty,
            max_tokens: SamplingParams::default().max_tokens,
        };

        Ok(Sampler::new(params, self.seed))
    }
}

impl Default for SamplerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for the [`OxiBonsaiConfig`].
pub struct ConfigBuilder {
    config: OxiBonsaiConfig,
}

impl ConfigBuilder {
    /// Create a new config builder with default values.
    pub fn new() -> Self {
        Self {
            config: OxiBonsaiConfig::default(),
        }
    }

    /// Set the path to the GGUF model file.
    pub fn model_path(mut self, path: impl Into<String>) -> Self {
        self.config.model.model_path = Some(path.into());
        self
    }

    /// Set the path to the tokenizer.json file.
    pub fn tokenizer_path(mut self, path: impl Into<String>) -> Self {
        self.config.model.tokenizer_path = Some(path.into());
        self
    }

    /// Set the maximum sequence length (prompt + generated).
    pub fn max_seq_len(mut self, len: usize) -> Self {
        self.config.model.max_seq_len = len;
        self
    }

    /// Set the server bind host address.
    pub fn host(mut self, h: impl Into<String>) -> Self {
        self.config.server.host = h.into();
        self
    }

    /// Set the server bind port.
    pub fn port(mut self, p: u16) -> Self {
        self.config.server.port = p;
        self
    }

    /// Set the log level filter (e.g. "info", "debug", "warn").
    pub fn log_level(mut self, level: impl Into<String>) -> Self {
        self.config.observability.log_level = level.into();
        self
    }

    /// Enable or disable JSON-formatted logs.
    pub fn json_logs(mut self, enabled: bool) -> Self {
        self.config.observability.json_logs = enabled;
        self
    }

    /// Set the sampling temperature.
    pub fn temperature(mut self, t: f32) -> Self {
        self.config.sampling.temperature = t;
        self
    }

    /// Set the top-k sampling parameter.
    pub fn top_k(mut self, k: usize) -> Self {
        self.config.sampling.top_k = k;
        self
    }

    /// Set the top-p (nucleus) sampling parameter.
    pub fn top_p(mut self, p: f32) -> Self {
        self.config.sampling.top_p = p;
        self
    }

    /// Set the repetition penalty.
    pub fn repetition_penalty(mut self, rp: f32) -> Self {
        self.config.sampling.repetition_penalty = rp;
        self
    }

    /// Set the maximum tokens to generate.
    pub fn max_tokens(mut self, n: usize) -> Self {
        self.config.sampling.max_tokens = n;
        self
    }

    /// Validate and build the [`OxiBonsaiConfig`].
    pub fn build(self) -> RuntimeResult<OxiBonsaiConfig> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl Default for ConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for the inference engine (the high-level entry point).
///
/// Validates configuration and sampling parameters together.
/// [`build`](Self::build) returns the validated config and sampler;
/// [`build_engine`](Self::build_engine) goes on to construct the real
/// [`InferenceEngine`](crate::engine::InferenceEngine) from a parsed GGUF,
/// honouring the configured `--rope-scaling` mode.
pub struct EngineBuilder {
    config: Option<OxiBonsaiConfig>,
    sampler: Option<SamplerBuilder>,
    kernel_tier: Option<String>,
    /// An explicit [`Self::rope_scaling`] call; `None` defers to the
    /// configuration's own `[model].rope_scaling`.
    rope_scaling: Option<RopeScalingMode>,
}

impl EngineBuilder {
    /// Create a new engine builder.
    pub fn new() -> Self {
        Self {
            config: None,
            sampler: None,
            kernel_tier: None,
            rope_scaling: None,
        }
    }

    /// Set the configuration directly.
    pub fn config(mut self, config: OxiBonsaiConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Load configuration from a TOML file.
    pub fn config_file(mut self, path: &str) -> RuntimeResult<Self> {
        let config = OxiBonsaiConfig::load(std::path::Path::new(path))?;
        self.config = Some(config);
        Ok(self)
    }

    /// Set a custom sampler builder.
    pub fn sampler(mut self, builder: SamplerBuilder) -> Self {
        self.sampler = Some(builder);
        self
    }

    /// Set the preferred kernel tier (e.g. "reference", "avx2", "neon").
    pub fn kernel_tier(mut self, tier: &str) -> Self {
        self.kernel_tier = Some(tier.to_string());
        self
    }

    /// Get the configured kernel tier name, if any.
    pub fn configured_kernel_tier(&self) -> Option<&str> {
        self.kernel_tier.as_deref()
    }

    /// Set the `--rope-scaling` override (M-08;
    /// see [`RopeScalingMode`]). It wins over the configuration's own
    /// `[model].rope_scaling`, is written into the configuration
    /// [`build`](Self::build) returns, and is applied to the model by
    /// [`build_engine`](Self::build_engine) (through
    /// `InferenceEngine::from_gguf_with_backend_and_rope`).
    pub fn rope_scaling(mut self, mode: RopeScalingMode) -> Self {
        self.rope_scaling = Some(mode);
        self
    }

    /// The RoPE-scaling mode an engine built by this builder uses: an
    /// explicit [`Self::rope_scaling`], else the configuration's
    /// `[model].rope_scaling`, else `auto`.
    pub fn configured_rope_scaling(&self) -> RopeScalingMode {
        self.rope_scaling
            .or_else(|| self.config.as_ref().map(|c| c.model.rope_scaling))
            .unwrap_or_default()
    }

    /// Validate and build the config + sampler pair.
    ///
    /// Returns the validated configuration and sampler, ready for
    /// engine construction once a GGUF file is available.
    pub fn build(self) -> RuntimeResult<(OxiBonsaiConfig, Sampler)> {
        let mut config = self.config.unwrap_or_default();
        if let Some(mode) = self.rope_scaling {
            config.model.rope_scaling = mode;
        }
        config.validate()?;

        let sampler = match self.sampler {
            Some(builder) => builder.build()?,
            None => {
                // Build sampler from config's sampling parameters
                SamplerBuilder::new()
                    .temperature(config.sampling.temperature)
                    .top_k(config.sampling.top_k)
                    .top_p(config.sampling.top_p)
                    .repetition_penalty(config.sampling.repetition_penalty)
                    .build()?
            }
        };

        Ok((config, sampler))
    }

    /// [`build`](Self::build), then construct the engine from `gguf` with
    /// the validated sampling parameters, `seed`, the configuration's
    /// `max_seq_len`, `backend`, and the configured RoPE-scaling mode.
    ///
    /// # Errors
    ///
    /// A configuration/sampler validation error, `--rope-scaling on` on a
    /// file that declares no scaling, or an engine construction error.
    pub fn build_engine<'a>(
        self,
        gguf: &'a oxibonsai_core::gguf::reader::GgufFile<'a>,
        seed: u64,
        backend: crate::engine_seam::Backend,
    ) -> RuntimeResult<(crate::engine::InferenceEngine<'a>, OxiBonsaiConfig)> {
        let (config, sampler) = self.build()?;
        let engine = crate::engine::InferenceEngine::from_gguf_with_backend_and_rope(
            gguf,
            sampler.params().clone(),
            seed,
            config.model.max_seq_len,
            backend,
            config.model.rope_scaling.into(),
        )?;
        Ok((engine, config))
    }
}

impl Default for EngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── SamplerBuilder tests ──

    #[test]
    fn sampler_builder_defaults() {
        let sampler = SamplerBuilder::new().build();
        assert!(sampler.is_ok());
        let sampler = sampler.expect("default build should succeed");
        let params = sampler.params();
        assert!((params.temperature - 0.7).abs() < f32::EPSILON);
        assert_eq!(params.top_k, 40);
        assert!((params.top_p - 0.9).abs() < f32::EPSILON);
        // The raw builder default is `1.0`
        // (no-op), matching `sampling::SamplingParams::default()`.
        assert!((params.repetition_penalty - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn sampler_builder_chain() {
        let sampler = SamplerBuilder::new()
            .temperature(0.5)
            .top_k(50)
            .top_p(0.95)
            .repetition_penalty(1.2)
            .seed(123)
            .build();
        assert!(sampler.is_ok());
        let sampler = sampler.expect("chained build should succeed");
        let params = sampler.params();
        assert!((params.temperature - 0.5).abs() < f32::EPSILON);
        assert_eq!(params.top_k, 50);
        assert!((params.top_p - 0.95).abs() < f32::EPSILON);
        assert!((params.repetition_penalty - 1.2).abs() < f32::EPSILON);
    }

    #[test]
    fn sampler_builder_negative_temperature() {
        let result = SamplerBuilder::new().temperature(-0.1).build();
        assert!(result.is_err());
        let err = result.expect_err("negative temperature should fail");
        assert!(err.to_string().contains("temperature"));
    }

    #[test]
    fn sampler_builder_invalid_top_p_high() {
        let result = SamplerBuilder::new().top_p(1.5).build();
        assert!(result.is_err());
        let err = result.expect_err("top_p > 1 should fail");
        assert!(err.to_string().contains("top_p"));
    }

    #[test]
    fn sampler_builder_invalid_top_p_low() {
        let result = SamplerBuilder::new().top_p(-0.1).build();
        assert!(result.is_err());
    }

    #[test]
    fn sampler_builder_invalid_repetition_penalty() {
        let result = SamplerBuilder::new().repetition_penalty(0.5).build();
        assert!(result.is_err());
        let err = result.expect_err("rep_pen < 1 should fail");
        assert!(err.to_string().contains("repetition_penalty"));
    }

    #[test]
    fn sampler_builder_zero_temperature() {
        let result = SamplerBuilder::new().temperature(0.0).build();
        assert!(result.is_ok());
    }

    #[test]
    fn sampler_builder_boundary_top_p() {
        // top_p = 0.0 and 1.0 should both be valid
        assert!(SamplerBuilder::new().top_p(0.0).build().is_ok());
        assert!(SamplerBuilder::new().top_p(1.0).build().is_ok());
    }

    #[test]
    fn sampler_builder_default_trait() {
        let builder = SamplerBuilder::default();
        assert!(builder.build().is_ok());
    }

    // ── ConfigBuilder tests ──

    #[test]
    fn config_builder_defaults() {
        let config = ConfigBuilder::new().build();
        assert!(config.is_ok());
        let config = config.expect("default build should succeed");
        assert_eq!(config.server.host, "0.0.0.0");
        assert_eq!(config.server.port, 8080);
        assert_eq!(config.model.max_seq_len, 4096);
    }

    #[test]
    fn config_builder_chain() {
        let model_path = std::env::temp_dir().join("model.gguf");
        let tokenizer_path = std::env::temp_dir().join("tokenizer.json");
        let config = ConfigBuilder::new()
            .model_path(model_path.display().to_string())
            .tokenizer_path(tokenizer_path.display().to_string())
            .max_seq_len(8192)
            .host("127.0.0.1")
            .port(3000)
            .log_level("debug")
            .json_logs(true)
            .temperature(0.5)
            .top_k(50)
            .top_p(0.95)
            .repetition_penalty(1.2)
            .max_tokens(1024)
            .build();
        assert!(config.is_ok());
        let config = config.expect("chained build should succeed");
        assert_eq!(
            config.model.model_path.as_deref(),
            Some(model_path.to_str().expect("path is valid UTF-8"))
        );
        assert_eq!(
            config.model.tokenizer_path.as_deref(),
            Some(tokenizer_path.to_str().expect("path is valid UTF-8"))
        );
        assert_eq!(config.model.max_seq_len, 8192);
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.server.port, 3000);
        assert_eq!(config.observability.log_level, "debug");
        assert!(config.observability.json_logs);
        assert!((config.sampling.temperature - 0.5).abs() < f32::EPSILON);
        assert_eq!(config.sampling.top_k, 50);
        assert_eq!(config.sampling.max_tokens, 1024);
    }

    #[test]
    fn config_builder_invalid_temperature() {
        let result = ConfigBuilder::new().temperature(-1.0).build();
        assert!(result.is_err());
    }

    #[test]
    fn config_builder_invalid_top_p() {
        let result = ConfigBuilder::new().top_p(2.0).build();
        assert!(result.is_err());
    }

    #[test]
    fn config_builder_invalid_max_seq_len() {
        let result = ConfigBuilder::new().max_seq_len(0).build();
        assert!(result.is_err());
    }

    #[test]
    fn config_builder_default_trait() {
        let builder = ConfigBuilder::default();
        assert!(builder.build().is_ok());
    }

    // ── EngineBuilder tests ──

    #[test]
    fn engine_builder_defaults() {
        let result = EngineBuilder::new().build();
        assert!(result.is_ok());
        let (config, _sampler) = result.expect("default build should succeed");
        assert_eq!(config.server.port, 8080);
    }

    #[test]
    fn engine_builder_with_config() {
        let config = ConfigBuilder::new()
            .port(9090)
            .build()
            .expect("config build should succeed");
        let result = EngineBuilder::new().config(config).build();
        assert!(result.is_ok());
        let (config, _sampler) = result.expect("build with config should succeed");
        assert_eq!(config.server.port, 9090);
    }

    #[test]
    fn engine_builder_with_sampler() {
        let sampler_builder = SamplerBuilder::new().temperature(0.3).seed(99);
        let result = EngineBuilder::new().sampler(sampler_builder).build();
        assert!(result.is_ok());
        let (_config, sampler) = result.expect("build with sampler should succeed");
        assert!((sampler.params().temperature - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn engine_builder_with_kernel_tier() {
        let builder = EngineBuilder::new().kernel_tier("reference");
        assert_eq!(builder.configured_kernel_tier(), Some("reference"));
        let result = builder.build();
        assert!(result.is_ok());
    }

    #[test]
    fn engine_builder_invalid_sampler() {
        let sampler_builder = SamplerBuilder::new().temperature(-1.0);
        let result = EngineBuilder::new().sampler(sampler_builder).build();
        assert!(result.is_err());
    }

    #[test]
    fn engine_builder_config_file_nonexistent() {
        let path = std::env::temp_dir().join("nonexistent_oxibonsai_test_12345.toml");
        let result = EngineBuilder::new().config_file(path.to_str().expect("path is valid UTF-8"));
        assert!(result.is_err());
    }

    #[test]
    fn engine_builder_config_file_valid() {
        let dir = std::env::temp_dir();
        let path = dir.join("oxibonsai_builder_test.toml");
        std::fs::write(
            &path,
            r#"
[server]
port = 7777
"#,
        )
        .expect("write temp config");

        let path_str = path.to_string_lossy().to_string();
        let result = EngineBuilder::new()
            .config_file(&path_str)
            .expect("should load config file")
            .build();
        assert!(result.is_ok());
        let (config, _) = result.expect("build should succeed");
        assert_eq!(config.server.port, 7777);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn engine_builder_rope_scaling_defaults_to_auto() {
        let builder = EngineBuilder::new();
        assert_eq!(builder.configured_rope_scaling(), RopeScalingMode::Auto);
    }

    #[test]
    fn engine_builder_rope_scaling_is_settable_and_survives_build() {
        let builder = EngineBuilder::new().rope_scaling(RopeScalingMode::Off);
        assert_eq!(builder.configured_rope_scaling(), RopeScalingMode::Off);
        assert!(builder.build().is_ok());
    }

    #[test]
    fn engine_builder_default_trait() {
        let builder = EngineBuilder::default();
        assert!(builder.build().is_ok());
    }

    // ── --rope-scaling reaches the built config and the engine ──

    #[test]
    fn an_explicit_rope_scaling_is_written_into_the_built_config() {
        let (config, _) = EngineBuilder::new()
            .rope_scaling(RopeScalingMode::Off)
            .build()
            .expect("valid");
        assert_eq!(config.model.rope_scaling, RopeScalingMode::Off);

        // Without an explicit call the configuration's own value stands.
        let mut from_toml = OxiBonsaiConfig::default();
        from_toml.model.rope_scaling = RopeScalingMode::On;
        let builder = EngineBuilder::new().config(from_toml);
        assert_eq!(builder.configured_rope_scaling(), RopeScalingMode::On);
        let (config, _) = builder.build().expect("valid");
        assert_eq!(config.model.rope_scaling, RopeScalingMode::On);

        // ...and an explicit call beats it.
        let mut from_toml = OxiBonsaiConfig::default();
        from_toml.model.rope_scaling = RopeScalingMode::On;
        let builder = EngineBuilder::new()
            .config(from_toml)
            .rope_scaling(RopeScalingMode::Auto);
        assert_eq!(builder.configured_rope_scaling(), RopeScalingMode::Auto);
    }

    fn small_builder(
        config_mode: RopeScalingMode,
        explicit: Option<RopeScalingMode>,
    ) -> EngineBuilder {
        let mut config = OxiBonsaiConfig::default();
        config.model.max_seq_len = 64;
        config.model.rope_scaling = config_mode;
        let builder = EngineBuilder::new().config(config);
        match explicit {
            Some(mode) => builder.rope_scaling(mode),
            None => builder,
        }
    }

    #[test]
    fn build_engine_applies_the_configured_rope_scaling() {
        use crate::engine_pool::tests::{
            build_tiny_gguf_bytes_with, engine_rope_scaling, yarn_metadata,
        };
        use oxibonsai_core::config::RopeScaling;
        let bytes = build_tiny_gguf_bytes_with(yarn_metadata());
        let gguf = oxibonsai_core::gguf::reader::GgufFile::parse(&bytes).expect("parse");
        let backend = crate::engine_seam::Backend::Cpu;

        let (auto, config) = small_builder(RopeScalingMode::Auto, None)
            .build_engine(&gguf, 42, backend)
            .expect("auto");
        assert!(matches!(
            engine_rope_scaling(&auto),
            RopeScaling::Yarn { .. }
        ));
        assert_eq!(config.model.max_seq_len, 64);

        let (off, config) = small_builder(RopeScalingMode::Auto, Some(RopeScalingMode::Off))
            .build_engine(&gguf, 42, backend)
            .expect("explicit off");
        assert_eq!(engine_rope_scaling(&off), RopeScaling::None);
        assert_eq!(config.model.rope_scaling, RopeScalingMode::Off);

        let (config_off, _) = small_builder(RopeScalingMode::Off, None)
            .build_engine(&gguf, 42, backend)
            .expect("[model].rope_scaling = off");
        assert_eq!(engine_rope_scaling(&config_off), RopeScaling::None);

        let plain = build_tiny_gguf_bytes_with(Vec::new());
        let plain = oxibonsai_core::gguf::reader::GgufFile::parse(&plain).expect("parse");
        let refused = small_builder(RopeScalingMode::Auto, Some(RopeScalingMode::On))
            .build_engine(&plain, 42, backend);
        assert!(
            refused.is_err(),
            "`on` refuses a file that declares no scaling"
        );
    }
}
