//! Per-layer profiling stats collected by `forward_with_stats`.

/// Statistics collected during a single layer's forward pass.
#[derive(Debug, Clone)]
pub struct LayerStats {
    /// Layer index.
    pub layer_idx: usize,
    /// Time spent on attention norm + Q/K/V projection.
    pub projection_us: u64,
    /// Time spent on RoPE application.
    pub rope_us: u64,
    /// Time spent on attention computation (GQA).
    pub attention_us: u64,
    /// Time spent on FFN (MLP) sublayer.
    pub ffn_us: u64,
    /// Total forward time for this layer.
    pub total_us: u64,
}

impl LayerStats {
    /// Create empty stats for a given layer.
    pub(crate) fn new(layer_idx: usize) -> Self {
        Self {
            layer_idx,
            projection_us: 0,
            rope_us: 0,
            attention_us: 0,
            ffn_us: 0,
            total_us: 0,
        }
    }

    /// Fraction of time spent in attention (vs total).
    pub fn attention_fraction(&self) -> f64 {
        if self.total_us == 0 {
            return 0.0;
        }
        self.attention_us as f64 / self.total_us as f64
    }

    /// Fraction of time spent in FFN (vs total).
    pub fn ffn_fraction(&self) -> f64 {
        if self.total_us == 0 {
            return 0.0;
        }
        self.ffn_us as f64 / self.total_us as f64
    }
}
