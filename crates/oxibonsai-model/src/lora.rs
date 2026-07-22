//! LoRA (Low-Rank Adaptation) adapter support for Bonsai/Qwen3 models.
//!
//! LoRA adds low-rank matrices A and B to existing weight matrices W.
//! The adapted output is: `W_adapted = W + (alpha/r) * B * A`
//! where A is `(r × d_in)` and B is `(d_out × r)`.
//!
//! ## Mathematical Background
//!
//! For a linear layer `y = Wx`, LoRA adds a bypass:
//! `y_adapted = Wx + (alpha/r) * B * A * x`
//!
//! - `A` is initialized with Kaiming uniform (He init) for stable training start
//! - `B` is initialized to zeros so the adapter has no effect at initialization
//! - `scaling = alpha / rank` controls contribution magnitude
//!
//! ## Typical Usage
//!
//! ```rust
//! use oxibonsai_model::lora::{LoraConfig, BonsaiLoraSet};
//!
//! let config = LoraConfig::default();
//! let mut lora_set = BonsaiLoraSet::new(config, "bonsai-8b", "my-adapter");
//! lora_set.add_attention_adapters(4096, 32);
//! lora_set.add_mlp_adapters(4096, 11008);
//! println!("Trainable params: {}", lora_set.total_trainable_params());
//! ```

use std::collections::HashMap;

// ──────────────────────────────────────────────────────────────────
// Errors
// ──────────────────────────────────────────────────────────────────

/// Errors produced by [`LoraAdapter`] operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoraError {
    /// A supplied buffer's length did not match the adapter's expected
    /// dimension.
    ///
    /// `context` names which buffer/dimension failed the check (e.g.
    /// `"apply: input length"` or `"merge_into_weights: weights length"`).
    DimensionMismatch {
        /// Human-readable description of what was being validated.
        context: &'static str,
        /// The length the adapter's dimensions require.
        expected: usize,
        /// The length actually supplied.
        got: usize,
    },
}

impl std::fmt::Display for LoraError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch {
                context,
                expected,
                got,
            } => write!(
                f,
                "LoRA dimension mismatch ({context}): expected length {expected}, got {got}"
            ),
        }
    }
}

impl std::error::Error for LoraError {}

// ──────────────────────────────────────────────────────────────────
// Configuration
// ──────────────────────────────────────────────────────────────────

/// LoRA adapter configuration controlling rank, scaling, and target layers.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LoraConfig {
    /// Rank `r` of the low-rank decomposition (typically 4, 8, 16, or 32).
    pub rank: usize,
    /// Alpha scaling factor (typically equal to rank).
    pub alpha: f32,
    /// Dropout probability during training (0.0 = no dropout, for inference).
    pub dropout: f32,
    /// Names of modules to adapt (e.g., `["q_proj", "v_proj"]`).
    pub target_modules: Vec<String>,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 8.0,
            dropout: 0.0,
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
                "gate_proj".to_string(),
                "up_proj".to_string(),
                "down_proj".to_string(),
            ],
        }
    }
}

// ──────────────────────────────────────────────────────────────────
// Single LoRA Adapter
// ──────────────────────────────────────────────────────────────────

/// A single LoRA adapter for one weight matrix.
///
/// Stores matrices A `(rank × d_in)` and B `(d_out × rank)` in row-major order.
/// The effective weight delta is `(alpha/rank) * B * A`.
pub struct LoraAdapter {
    /// Configuration used to create this adapter.
    pub config: LoraConfig,
    /// A matrix of shape `(rank × d_in)`, row-major.
    pub a_matrix: Vec<f32>,
    /// B matrix of shape `(d_out × rank)`, row-major.
    pub b_matrix: Vec<f32>,
    /// Input dimension.
    pub d_in: usize,
    /// Output dimension.
    pub d_out: usize,
    /// Precomputed `alpha / rank` scaling factor.
    pub scaling: f32,
}

impl LoraAdapter {
    /// Create a new LoRA adapter.
    ///
    /// - A is initialized with Kaiming uniform (He init): uniform in `[-sqrt(2/d_in), sqrt(2/d_in)]`
    /// - B is initialized to all zeros (no effect at init)
    ///
    /// Uses a deterministic xorshift64 PRNG seeded from the dimensions,
    /// so no external randomness crate is needed.
    pub fn new(d_in: usize, d_out: usize, config: LoraConfig) -> Self {
        let rank = config.rank;
        let scaling = if rank > 0 {
            config.alpha / rank as f32
        } else {
            0.0
        };
        let a_size = rank * d_in;
        let b_size = d_out * rank;

        // He/Kaiming uniform bound: sqrt(2 / fan_in)
        let bound = if d_in > 0 {
            (2.0_f32 / d_in as f32).sqrt()
        } else {
            0.0
        };

        // Deterministic xorshift64 PRNG seeded from dimensions
        let mut rng_state: u64 = (d_in as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(d_out as u64)
            .wrapping_add(rank as u64)
            .wrapping_add(1);

        let mut next_f32 = move || -> f32 {
            // xorshift64
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            // Map to [-bound, bound]
            let t = (rng_state as f32) / (u64::MAX as f32);
            t * 2.0 * bound - bound
        };

        let a_matrix: Vec<f32> = (0..a_size).map(|_| next_f32()).collect();
        let b_matrix = vec![0.0f32; b_size];

        Self {
            config,
            a_matrix,
            b_matrix,
            d_in,
            d_out,
            scaling,
        }
    }

    /// Apply the adapter to input vector `x`.
    ///
    /// Computes `(B * A * x) * scaling`.
    /// - `x` has shape `[d_in]`
    /// - Returns vector of shape `[d_out]`
    ///
    /// This output should be *added* to the base layer output.
    ///
    /// # Errors
    ///
    /// Returns [`LoraError::DimensionMismatch`] if `x.len() != self.d_in`.
    /// Without this check a shorter-than-expected `x` would silently sum over
    /// fewer terms than the adapter was trained for, producing a
    /// plausible-looking but numerically wrong result with no error signal.
    pub fn apply(&self, x: &[f32]) -> Result<Vec<f32>, LoraError> {
        if x.len() != self.d_in {
            return Err(LoraError::DimensionMismatch {
                context: "apply: input length",
                expected: self.d_in,
                got: x.len(),
            });
        }

        let rank = self.config.rank;

        // Step 1: intermediate = A * x  (shape: [rank])
        let mut intermediate = vec![0.0f32; rank];
        for (r, slot) in intermediate.iter_mut().enumerate().take(rank) {
            let row_offset = r * self.d_in;
            let mut sum = 0.0f32;
            for (j, &xj) in x.iter().enumerate().take(self.d_in) {
                sum += self.a_matrix[row_offset + j] * xj;
            }
            *slot = sum;
        }

        // Step 2: output = B * intermediate  (shape: [d_out])
        let mut output = vec![0.0f32; self.d_out];
        for (i, slot) in output.iter_mut().enumerate().take(self.d_out) {
            let row_offset = i * rank;
            let mut sum = 0.0f32;
            for (r, &inter_r) in intermediate.iter().enumerate().take(rank) {
                sum += self.b_matrix[row_offset + r] * inter_r;
            }
            *slot = sum * self.scaling;
        }

        Ok(output)
    }

    /// Total number of trainable parameters: `(d_in * rank) + (rank * d_out)`.
    pub fn param_count(&self) -> usize {
        self.d_in * self.config.rank + self.config.rank * self.d_out
    }

    /// Memory used by this adapter in bytes (both matrices as f32).
    pub fn memory_bytes(&self) -> usize {
        self.param_count() * std::mem::size_of::<f32>()
    }

    /// Fold this adapter permanently into the base weight matrix.
    ///
    /// Computes `W += (alpha/rank) * B * A` in-place on `weights`.
    /// `weights` must be row-major with shape `[d_out × d_in]`.
    ///
    /// After merging, this adapter is no longer needed for inference.
    ///
    /// # Errors
    ///
    /// Returns [`LoraError::DimensionMismatch`] if
    /// `weights.len() != d_out * d_in`. Without this check an undersized
    /// buffer would index past its end and panic instead of returning a
    /// typed error.
    pub fn merge_into_weights(&self, weights: &mut [f32]) -> Result<(), LoraError> {
        let expected_len = self.d_out * self.d_in;
        if weights.len() != expected_len {
            return Err(LoraError::DimensionMismatch {
                context: "merge_into_weights: weights length",
                expected: expected_len,
                got: weights.len(),
            });
        }

        let rank = self.config.rank;
        for i in 0..self.d_out {
            for j in 0..self.d_in {
                let mut delta = 0.0f32;
                for r in 0..rank {
                    delta += self.b_matrix[i * rank + r] * self.a_matrix[r * self.d_in + j];
                }
                weights[i * self.d_in + j] += self.scaling * delta;
            }
        }
        Ok(())
    }
}

// ──────────────────────────────────────────────────────────────────
// Registry of LoRA Adapters
// ──────────────────────────────────────────────────────────────────

/// Registry holding all LoRA adapters for a model, keyed by module name.
pub struct LoraRegistry {
    adapters: HashMap<String, LoraAdapter>,
    config: LoraConfig,
}

impl LoraRegistry {
    /// Create an empty registry with the given configuration.
    pub fn new(config: LoraConfig) -> Self {
        Self {
            adapters: HashMap::new(),
            config,
        }
    }

    /// Register an adapter for the named module.
    pub fn add(&mut self, module_name: &str, adapter: LoraAdapter) {
        self.adapters.insert(module_name.to_string(), adapter);
    }

    /// Retrieve the adapter for a module by name.
    pub fn get(&self, module_name: &str) -> Option<&LoraAdapter> {
        self.adapters.get(module_name)
    }

    /// Apply the named adapter to input vector `x`.
    ///
    /// Returns `None` if no adapter is registered for that module, or
    /// `Some(Err(LoraError::DimensionMismatch))` if `x.len()` does not match
    /// the registered adapter's `d_in`.
    pub fn apply_adapter(
        &self,
        module_name: &str,
        x: &[f32],
    ) -> Option<Result<Vec<f32>, LoraError>> {
        self.adapters.get(module_name).map(|a| a.apply(x))
    }

    /// Number of registered adapters.
    pub fn adapter_count(&self) -> usize {
        self.adapters.len()
    }

    /// Total trainable parameter count across all adapters.
    pub fn total_param_count(&self) -> usize {
        self.adapters.values().map(|a| a.param_count()).sum()
    }

    /// Total memory consumed by all adapters in bytes.
    pub fn total_memory_bytes(&self) -> usize {
        self.adapters.values().map(|a| a.memory_bytes()).sum()
    }

    /// Returns `true` if no adapters have been registered.
    pub fn is_empty(&self) -> bool {
        self.adapters.is_empty()
    }

    /// Names of all registered adapter modules.
    pub fn module_names(&self) -> Vec<&str> {
        self.adapters.keys().map(|s| s.as_str()).collect()
    }

    /// Reference to the registry-level configuration.
    pub fn config(&self) -> &LoraConfig {
        &self.config
    }

    /// Estimate how many adapters of given dimensions fit within a memory budget.
    ///
    /// Returns 0 if a single adapter exceeds the budget or dimensions are zero.
    pub fn estimate_adapters_for_budget(
        d_in: usize,
        d_out: usize,
        rank: usize,
        budget_bytes: usize,
    ) -> usize {
        let params_per_adapter = d_in * rank + rank * d_out;
        let bytes_per_adapter = params_per_adapter * std::mem::size_of::<f32>();
        if bytes_per_adapter == 0 {
            return 0;
        }
        budget_bytes / bytes_per_adapter
    }
}

// ──────────────────────────────────────────────────────────────────
// BonsaiLoraSet — high-level adapter set for Qwen3/Bonsai models
// ──────────────────────────────────────────────────────────────────

/// A complete LoRA adapter set for a specific Bonsai/Qwen3 model.
///
/// Manages the standard Qwen3 projection adapters:
/// `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
pub struct BonsaiLoraSet {
    /// The underlying adapter registry.
    pub registry: LoraRegistry,
    /// Name of the base model this adapter targets.
    pub base_model_name: String,
    /// Identifier for this specific adapter.
    pub adapter_name: String,
}

impl BonsaiLoraSet {
    /// Create a new (empty) LoRA set for the named base model.
    pub fn new(config: LoraConfig, model_name: &str, adapter_name: &str) -> Self {
        Self {
            registry: LoraRegistry::new(config),
            base_model_name: model_name.to_string(),
            adapter_name: adapter_name.to_string(),
        }
    }

    /// Add attention projection adapters: `q_proj`, `k_proj`, `v_proj`, `o_proj`.
    ///
    /// - `hidden_size`: model hidden dimension (e.g., 4096 for 8B)
    /// - `num_heads`: number of query attention heads
    ///
    /// Dimensions follow Qwen3 GQA conventions:
    /// - `q_proj`: `hidden_size → hidden_size`
    /// - `k_proj`, `v_proj`: `hidden_size → head_dim * num_heads` (full head projection)
    /// - `o_proj`: `hidden_size → hidden_size`
    pub fn add_attention_adapters(&mut self, hidden_size: usize, num_heads: usize) {
        let head_dim = hidden_size.checked_div(num_heads).unwrap_or(hidden_size);
        let kv_dim = head_dim * num_heads;
        let config = self.registry.config.clone();

        self.registry.add(
            "q_proj",
            LoraAdapter::new(hidden_size, hidden_size, config.clone()),
        );
        self.registry.add(
            "k_proj",
            LoraAdapter::new(hidden_size, kv_dim, config.clone()),
        );
        self.registry.add(
            "v_proj",
            LoraAdapter::new(hidden_size, kv_dim, config.clone()),
        );
        self.registry
            .add("o_proj", LoraAdapter::new(hidden_size, hidden_size, config));
    }

    /// Add MLP projection adapters: `gate_proj`, `up_proj`, `down_proj`.
    ///
    /// - `hidden_size`: model hidden dimension
    /// - `intermediate_size`: FFN intermediate dimension (e.g., 11008 for 8B)
    pub fn add_mlp_adapters(&mut self, hidden_size: usize, intermediate_size: usize) {
        let config = self.registry.config.clone();

        self.registry.add(
            "gate_proj",
            LoraAdapter::new(hidden_size, intermediate_size, config.clone()),
        );
        self.registry.add(
            "up_proj",
            LoraAdapter::new(hidden_size, intermediate_size, config.clone()),
        );
        self.registry.add(
            "down_proj",
            LoraAdapter::new(intermediate_size, hidden_size, config),
        );
    }

    /// Total number of trainable LoRA parameters across all registered adapters.
    pub fn total_trainable_params(&self) -> usize {
        self.registry.total_param_count()
    }

    /// Ratio of LoRA parameters to base model parameters.
    ///
    /// Typically 0.1% to 1% for well-configured LoRA adapters.
    /// Returns 0.0 if `base_param_count` is zero.
    pub fn efficiency_ratio(&self, base_param_count: u64) -> f32 {
        if base_param_count == 0 {
            return 0.0;
        }
        self.total_trainable_params() as f32 / base_param_count as f32
    }
}

// ──────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(rank: usize) -> LoraConfig {
        LoraConfig {
            rank,
            alpha: rank as f32,
            dropout: 0.0,
            target_modules: vec![],
        }
    }

    #[test]
    fn test_lora_adapter_apply_zero_b_matrix() {
        // B is initialized to zeros, so output must be all zeros regardless of A or x.
        let config = make_config(4);
        let adapter = LoraAdapter::new(8, 16, config);
        let x = vec![1.0f32; 8];
        let out = adapter.apply(&x).expect("apply should succeed");
        assert_eq!(out.len(), 16, "output must have d_out elements");
        for (i, v) in out.iter().enumerate() {
            assert!(
                v.abs() < 1e-6,
                "output[{i}] = {v} but B is zero so output must be zero"
            );
        }
    }

    #[test]
    fn test_lora_adapter_apply_undersized_input_errors() {
        // x shorter than d_in must return a typed error instead of silently
        // summing over fewer terms and returning a wrong-but-plausible result.
        let config = make_config(4);
        let adapter = LoraAdapter::new(8, 16, config);
        let x = vec![1.0f32; 5]; // too short: d_in = 8
        let result = adapter.apply(&x);
        match result {
            Err(LoraError::DimensionMismatch { expected, got, .. }) => {
                assert_eq!(expected, 8);
                assert_eq!(got, 5);
            }
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn test_lora_adapter_apply_oversized_input_errors() {
        let config = make_config(4);
        let adapter = LoraAdapter::new(8, 16, config);
        let x = vec![1.0f32; 12]; // too long: d_in = 8
        let result = adapter.apply(&x);
        match result {
            Err(LoraError::DimensionMismatch { expected, got, .. }) => {
                assert_eq!(expected, 8);
                assert_eq!(got, 12);
            }
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn test_lora_adapter_param_count() {
        let adapter = LoraAdapter::new(64, 128, make_config(8));
        // (d_in * rank) + (rank * d_out) = 64*8 + 8*128 = 512 + 1024 = 1536
        assert_eq!(adapter.param_count(), 64 * 8 + 8 * 128);
    }

    #[test]
    fn test_lora_adapter_memory_bytes() {
        let adapter = LoraAdapter::new(64, 128, make_config(8));
        // Each f32 is 4 bytes
        assert_eq!(adapter.memory_bytes(), adapter.param_count() * 4);
    }

    #[test]
    fn test_lora_registry_add_and_get() {
        let mut registry = LoraRegistry::new(make_config(4));
        let adapter = LoraAdapter::new(32, 64, make_config(4));
        registry.add("q_proj", adapter);

        assert_eq!(
            registry.adapter_count(),
            1,
            "one adapter should be registered"
        );
        assert!(
            registry.get("q_proj").is_some(),
            "q_proj should be retrievable"
        );
        assert!(registry.get("v_proj").is_none(), "v_proj should not exist");
        assert!(!registry.is_empty());
        let names = registry.module_names();
        assert!(names.contains(&"q_proj"));
    }

    #[test]
    fn test_lora_registry_apply_adapter() {
        let mut registry = LoraRegistry::new(make_config(4));
        let adapter = LoraAdapter::new(8, 16, make_config(4));
        registry.add("q_proj", adapter);

        let x = vec![0.5f32; 8];
        let out = registry.apply_adapter("q_proj", &x);
        assert!(
            out.is_some(),
            "apply should return Some for registered module"
        );
        let out_vec = out
            .expect("output must be Some")
            .expect("apply should succeed for matching dimensions");
        assert_eq!(out_vec.len(), 16);

        // Non-existent module returns None
        assert!(registry.apply_adapter("missing", &x).is_none());
    }

    #[test]
    fn test_lora_registry_apply_adapter_dimension_mismatch() {
        let mut registry = LoraRegistry::new(make_config(4));
        let adapter = LoraAdapter::new(8, 16, make_config(4));
        registry.add("q_proj", adapter);

        let x = vec![0.5f32; 3]; // wrong length: d_in = 8
        let out = registry.apply_adapter("q_proj", &x);
        match out {
            Some(Err(LoraError::DimensionMismatch { expected, got, .. })) => {
                assert_eq!(expected, 8);
                assert_eq!(got, 3);
            }
            other => panic!("expected Some(Err(DimensionMismatch)), got {other:?}"),
        }
    }

    #[test]
    fn test_bonsai_lora_set_creates_adapters() {
        let config = make_config(8);
        let mut lora_set = BonsaiLoraSet::new(config, "bonsai-8b", "my-adapter");
        lora_set.add_attention_adapters(256, 8);
        lora_set.add_mlp_adapters(256, 512);

        // 4 attention adapters + 3 mlp adapters = 7 total
        assert_eq!(
            lora_set.registry.adapter_count(),
            7,
            "should have 7 adapters (4 attn + 3 mlp)"
        );
        assert_eq!(lora_set.base_model_name, "bonsai-8b");
        assert_eq!(lora_set.adapter_name, "my-adapter");
    }

    #[test]
    fn test_lora_merge_into_weights() {
        // B is all zeros at init, so merging should not change any weights.
        let config = make_config(4);
        let adapter = LoraAdapter::new(4, 4, config);
        let mut weights = vec![1.0f32; 16]; // 4×4 identity-like
        adapter
            .merge_into_weights(&mut weights)
            .expect("merge_into_weights should succeed for matching dimensions");
        for (i, w) in weights.iter().enumerate() {
            assert!(
                (w - 1.0).abs() < 1e-6,
                "weights[{i}] = {w}, expected 1.0 (B is zero so no change)"
            );
        }
    }

    #[test]
    fn test_lora_merge_into_weights_undersized_buffer_errors() {
        // weights shorter than d_out * d_in must return a typed error instead
        // of panicking with an out-of-bounds index.
        let config = make_config(4);
        let adapter = LoraAdapter::new(4, 4, config);
        let mut weights = vec![1.0f32; 10]; // needs 16 (4*4)
        let result = adapter.merge_into_weights(&mut weights);
        match result {
            Err(LoraError::DimensionMismatch { expected, got, .. }) => {
                assert_eq!(expected, 16);
                assert_eq!(got, 10);
            }
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn test_lora_merge_into_weights_oversized_buffer_errors() {
        let config = make_config(4);
        let adapter = LoraAdapter::new(4, 4, config);
        let mut weights = vec![1.0f32; 20]; // more than 16 (4*4)
        let result = adapter.merge_into_weights(&mut weights);
        match result {
            Err(LoraError::DimensionMismatch { expected, got, .. }) => {
                assert_eq!(expected, 16);
                assert_eq!(got, 20);
            }
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn test_lora_efficiency_ratio() {
        let config = make_config(8);
        let mut lora_set = BonsaiLoraSet::new(config, "bonsai-8b", "test");
        // Small dimensions to keep test fast
        lora_set.add_attention_adapters(256, 8);

        let base_params = 8_000_000_000u64; // 8B parameter model
        let ratio = lora_set.efficiency_ratio(base_params);
        assert!(ratio > 0.0, "ratio must be positive");
        assert!(
            ratio < 0.01,
            "LoRA should be <1% of base params, got ratio={ratio}"
        );

        // Edge case: zero base params
        assert_eq!(lora_set.efficiency_ratio(0), 0.0);
    }

    #[test]
    fn test_lora_estimate_adapters_for_budget() {
        // bytes per adapter = (64*8 + 8*128) * 4 = 1536 * 4 = 6144
        let bytes_per = (64 * 8 + 8 * 128) * 4;
        let budget = bytes_per * 5;
        let n = LoraRegistry::estimate_adapters_for_budget(64, 128, 8, budget);
        assert_eq!(
            n, 5,
            "should fit exactly 5 adapters in budget of 5×per-adapter"
        );

        // Zero budget
        assert_eq!(LoraRegistry::estimate_adapters_for_budget(64, 128, 8, 0), 0);
        // Zero dimensions
        assert_eq!(
            LoraRegistry::estimate_adapters_for_budget(0, 0, 0, 1_000_000),
            0
        );
    }
}
