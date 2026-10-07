//! Mixture-of-Experts (MoE) router — top-K gating with load balancing.
//!
//! This module implements:
//! - [`MoeConfig`] — configuration for the MoE layer
//! - [`RoutingDecision`] — result of routing a batch of tokens
//! - [`TopKRouter`] — learned gating that sends each token to top-K experts
//! - [`ExpertBuffer`] — per-expert token-slot management with capacity control
//! - [`combine_expert_outputs`] — weighted accumulation of expert outputs

use serde::{Deserialize, Serialize};

// ──────────────────────────────────────────────────────────────────
// Configuration
// ──────────────────────────────────────────────────────────────────

/// Configuration for a Mixture-of-Experts layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoeConfig {
    /// Total number of experts (e.g., 8 or 64).
    pub num_experts: usize,
    /// Number of experts activated per token (e.g., 2).
    pub top_k: usize,
    /// Hidden dimension size fed into the router.
    pub hidden_size: usize,
    /// Overflow buffer factor: capacity = ceil(capacity_factor * tokens / num_experts).
    pub capacity_factor: f32,
    /// Z-loss coefficient to prevent router entropy collapse (default 1e-3).
    pub z_loss_coeff: f32,
    /// Load-balancing auxiliary loss coefficient (default 1e-2).
    pub aux_loss_coeff: f32,
}

impl Default for MoeConfig {
    fn default() -> Self {
        Self {
            num_experts: 8,
            top_k: 2,
            hidden_size: 256,
            capacity_factor: 1.25,
            z_loss_coeff: 1e-3,
            aux_loss_coeff: 1e-2,
        }
    }
}

// ──────────────────────────────────────────────────────────────────
// Routing decision
// ──────────────────────────────────────────────────────────────────

/// The routing decision produced for a batch of tokens.
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// `[num_tokens][top_k]` — which experts each token is assigned to.
    pub expert_indices: Vec<Vec<usize>>,
    /// `[num_tokens][top_k]` — softmax weights for each assigned expert.
    pub expert_weights: Vec<Vec<f32>>,
    /// `[num_experts]` — how many tokens each expert receives.
    pub expert_load: Vec<usize>,
    /// `[num_tokens]` — true if the token was dropped due to expert over-capacity.
    pub overflow_mask: Vec<bool>,
    /// Load-balancing auxiliary loss value for this batch.
    pub aux_loss: f32,
    /// Z-loss (router entropy penalty) for this batch.
    pub z_loss: f32,
}

impl RoutingDecision {
    /// Number of tokens in this routing decision.
    pub fn num_tokens(&self) -> usize {
        self.expert_indices.len()
    }

    /// Load-balance score: 1 - coefficient_of_variation(load).
    ///
    /// A score of 1.0 means perfectly balanced; lower values indicate imbalance.
    pub fn load_balance_score(&self) -> f32 {
        let n = self.expert_load.len();
        if n == 0 {
            return 1.0;
        }
        let mean = self.expert_load.iter().sum::<usize>() as f32 / n as f32;
        if mean == 0.0 {
            return 1.0;
        }
        let variance = self
            .expert_load
            .iter()
            .map(|&l| {
                let diff = l as f32 - mean;
                diff * diff
            })
            .sum::<f32>()
            / n as f32;
        let std_dev = variance.sqrt();
        1.0 - (std_dev / mean)
    }

    /// Fraction of tokens that were dropped due to capacity overflow.
    pub fn overflow_rate(&self) -> f32 {
        let total = self.overflow_mask.len();
        if total == 0 {
            return 0.0;
        }
        let overflowed = self.overflow_mask.iter().filter(|&&b| b).count();
        overflowed as f32 / total as f32
    }

    /// Maximum load across all experts.
    pub fn max_expert_load(&self) -> usize {
        self.expert_load.iter().copied().max().unwrap_or(0)
    }

    /// Minimum load across all experts.
    pub fn min_expert_load(&self) -> usize {
        self.expert_load.iter().copied().min().unwrap_or(0)
    }
}

// ──────────────────────────────────────────────────────────────────
// Top-K router
// ──────────────────────────────────────────────────────────────────

/// Learned gating router that sends each token to the top-K experts.
pub struct TopKRouter {
    /// Router configuration.
    pub config: MoeConfig,
    /// Weight matrix `[num_experts × hidden_size]` (row-major).
    pub weights: Vec<f32>,
}

impl TopKRouter {
    /// Create a new router with LCG-initialised weights (no `rand` crate).
    pub fn new(config: MoeConfig) -> Self {
        let n = config.num_experts * config.hidden_size;
        let weights = lcg_init_weights(n, 0x1234_5678_9abc_def0u64);
        Self { config, weights }
    }

    /// Route a batch of token hidden states.
    ///
    /// `input` must have length `num_tokens * hidden_size`.
    /// Returns the full [`RoutingDecision`] for the batch.
    pub fn route(&self, input: &[f32], num_tokens: usize) -> RoutingDecision {
        let cfg = &self.config;
        let e = cfg.num_experts;
        let k = cfg.top_k;
        let h = cfg.hidden_size;

        debug_assert_eq!(
            input.len(),
            num_tokens * h,
            "input length mismatch: expected {} got {}",
            num_tokens * h,
            input.len()
        );

        // Step 1: logits = input @ weights.T  →  [num_tokens × num_experts]
        let mut logits = vec![0.0f32; num_tokens * e];
        for t in 0..num_tokens {
            let x = &input[t * h..(t + 1) * h];
            for ex in 0..e {
                let w = &self.weights[ex * h..(ex + 1) * h];
                let mut dot = 0.0f32;
                for i in 0..h {
                    dot += x[i] * w[i];
                }
                logits[t * e + ex] = dot;
            }
        }

        // Step 2: softmax(logits) per token  →  router_probs [num_tokens × num_experts]
        let mut router_probs = vec![0.0f32; num_tokens * e];
        for t in 0..num_tokens {
            let row = &logits[t * e..(t + 1) * e];
            let max_l = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_vals: Vec<f32> = row.iter().map(|&l| (l - max_l).exp()).collect();
            let sum: f32 = exp_vals.iter().sum();
            for (ex, &ev) in exp_vals.iter().enumerate() {
                router_probs[t * e + ex] = ev / sum.max(1e-9);
            }
        }

        // Step 3: top-K per token
        let mut top_k_indices: Vec<Vec<usize>> = Vec::with_capacity(num_tokens);
        let mut top_k_weights: Vec<Vec<f32>> = Vec::with_capacity(num_tokens);
        for t in 0..num_tokens {
            let probs = &router_probs[t * e..(t + 1) * e];
            let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
            // Sort descending by probability
            indexed.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            let chosen: Vec<usize> = indexed[..k].iter().map(|&(i, _)| i).collect();
            // Re-normalise weights over chosen experts only
            let chosen_sum: f32 = chosen.iter().map(|&i| probs[i]).sum();
            let chosen_w: Vec<f32> = chosen
                .iter()
                .map(|&i| probs[i] / chosen_sum.max(1e-9))
                .collect();
            top_k_indices.push(chosen);
            top_k_weights.push(chosen_w);
        }

        // Step 4: expert capacity = ceil(capacity_factor * num_tokens / num_experts)
        let capacity =
            ((cfg.capacity_factor * num_tokens as f32 / e as f32).ceil() as usize).max(1);

        // Step 5: assign tokens to experts; detect overflow
        let mut buffer = ExpertBuffer::new(e, capacity);
        let mut overflow_mask = vec![false; num_tokens];

        for t in 0..num_tokens {
            for &ex in &top_k_indices[t] {
                let ok = buffer.try_assign(t, ex);
                if !ok {
                    overflow_mask[t] = true;
                }
            }
        }

        let expert_load: Vec<usize> = (0..e).map(|ex| buffer.counts[ex]).collect();

        // Step 6: auxiliary losses
        let aux_loss_val =
            Self::aux_loss(&expert_load, &router_probs, num_tokens, cfg.aux_loss_coeff);
        let z_loss_val = Self::z_loss(&logits, num_tokens, e, cfg.z_loss_coeff);

        RoutingDecision {
            expert_indices: top_k_indices,
            expert_weights: top_k_weights,
            expert_load,
            overflow_mask,
            aux_loss: aux_loss_val,
            z_loss: z_loss_val,
        }
    }

    /// Z-loss: penalises large router logits to prevent expert collapse.
    ///
    /// z_loss = coeff * mean_t( log(sum_e(exp(logits_t)))^2 )
    pub fn z_loss(logits: &[f32], num_tokens: usize, num_experts: usize, coeff: f32) -> f32 {
        if num_tokens == 0 {
            return 0.0;
        }
        let mut total = 0.0f32;
        for t in 0..num_tokens {
            let row = &logits[t * num_experts..(t + 1) * num_experts];
            let max_l = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let log_sum_exp: f32 = max_l + row.iter().map(|&l| (l - max_l).exp()).sum::<f32>().ln();
            total += log_sum_exp * log_sum_exp;
        }
        coeff * total / num_tokens as f32
    }

    /// Auxiliary load-balancing loss.
    ///
    /// aux_loss = coeff * num_experts * sum_i(f_i * p_i)
    /// where f_i = fraction of tokens routed to expert i,
    ///       p_i = mean router probability for expert i.
    pub fn aux_loss(
        expert_load: &[usize],
        expert_probs: &[f32],
        num_tokens: usize,
        coeff: f32,
    ) -> f32 {
        let num_experts = expert_load.len();
        if num_tokens == 0 || num_experts == 0 {
            return 0.0;
        }

        // f_i: fraction of tokens assigned to expert i
        let f: Vec<f32> = expert_load
            .iter()
            .map(|&load| load as f32 / num_tokens as f32)
            .collect();

        // p_i: mean router probability for expert i across all tokens
        let mut p = vec![0.0f32; num_experts];
        for t in 0..num_tokens {
            for ex in 0..num_experts {
                p[ex] += expert_probs[t * num_experts + ex];
            }
        }
        for slot in p.iter_mut().take(num_experts) {
            *slot /= num_tokens as f32;
        }

        let sum: f32 = f.iter().zip(p.iter()).map(|(&fi, &pi)| fi * pi).sum();
        coeff * num_experts as f32 * sum
    }
}

// ──────────────────────────────────────────────────────────────────
// Expert buffer
// ──────────────────────────────────────────────────────────────────

/// Manages per-expert token slots with hard capacity limits.
pub struct ExpertBuffer {
    /// Maximum tokens per expert.
    pub capacity: usize,
    /// Number of experts.
    pub num_experts: usize,
    /// `[num_experts][capacity]` — token indices assigned to each expert slot.
    slots: Vec<Vec<usize>>,
    /// Current count of assigned tokens per expert.
    pub counts: Vec<usize>,
    /// Total tokens dropped due to capacity overflow.
    overflow_count: usize,
}

impl ExpertBuffer {
    /// Create a new empty buffer.
    pub fn new(num_experts: usize, capacity: usize) -> Self {
        Self {
            capacity,
            num_experts,
            slots: vec![Vec::with_capacity(capacity); num_experts],
            counts: vec![0usize; num_experts],
            overflow_count: 0,
        }
    }

    /// Try to assign `token_idx` to `expert_idx`.
    ///
    /// Returns `false` if the expert is already at capacity.
    pub fn try_assign(&mut self, token_idx: usize, expert_idx: usize) -> bool {
        debug_assert!(expert_idx < self.num_experts);
        if self.counts[expert_idx] >= self.capacity {
            self.overflow_count += 1;
            return false;
        }
        self.slots[expert_idx].push(token_idx);
        self.counts[expert_idx] += 1;
        true
    }

    /// Returns the token indices assigned to `expert_idx`.
    pub fn tokens_for_expert(&self, expert_idx: usize) -> &[usize] {
        &self.slots[expert_idx]
    }

    /// Fraction of capacity used by `expert_idx` (0.0 – 1.0).
    pub fn expert_utilization(&self, expert_idx: usize) -> f32 {
        if self.capacity == 0 {
            return 0.0;
        }
        self.counts[expert_idx] as f32 / self.capacity as f32
    }

    /// Total number of (token, expert) assignments successfully made.
    pub fn total_assigned(&self) -> usize {
        self.counts.iter().sum()
    }

    /// Total tokens dropped due to capacity overflow.
    pub fn overflow_count(&self) -> usize {
        self.overflow_count
    }

    /// Reset all slots and counts, ready to route a new batch.
    pub fn reset(&mut self) {
        for slot in &mut self.slots {
            slot.clear();
        }
        self.counts.fill(0);
        self.overflow_count = 0;
    }
}

// ──────────────────────────────────────────────────────────────────
// Output combination
// ──────────────────────────────────────────────────────────────────

/// Combine expert outputs weighted by router scores into a single output tensor.
///
/// # Arguments
/// - `expert_outputs`: `[num_experts]` slices, each `[tokens_for_expert × hidden_size]`.
///   The tokens in each slice appear in the order stored by the [`ExpertBuffer`]
///   (i.e., the same order as `routing.expert_indices` produces).
/// - `routing`: the [`RoutingDecision`] from the router.
/// - `num_tokens`: total number of input tokens.
/// - `hidden_size`: hidden dimension.
///
/// Returns `[num_tokens × hidden_size]`.
pub fn combine_expert_outputs(
    expert_outputs: &[Vec<f32>],
    routing: &RoutingDecision,
    num_tokens: usize,
    hidden_size: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; num_tokens * hidden_size];

    // For each expert, we need to know which token each slot corresponds to.
    // We reconstruct the per-expert slot → token mapping from routing.expert_indices.
    let num_experts = expert_outputs.len();
    // expert_slot_cursor[ex] tracks the next slot index to read from expert_outputs[ex]
    let mut slot_cursor = vec![0usize; num_experts];
    // Build per-expert list of (token_idx, weight) in slot order
    let mut expert_token_list: Vec<Vec<(usize, f32)>> = vec![Vec::new(); num_experts];

    for t in 0..num_tokens {
        if routing.overflow_mask[t] {
            // Overflowed tokens are skipped in expert processing
            continue;
        }
        for (rank, &ex) in routing.expert_indices[t].iter().enumerate() {
            let w = routing.expert_weights[t][rank];
            expert_token_list[ex].push((t, w));
        }
    }

    for ex in 0..num_experts {
        let ex_out = &expert_outputs[ex];
        for (slot, &(token_idx, weight)) in expert_token_list[ex].iter().enumerate() {
            let _ = slot_cursor[ex]; // unused beyond bookkeeping
            let src_start = slot * hidden_size;
            let dst_start = token_idx * hidden_size;
            if src_start + hidden_size > ex_out.len() {
                break;
            }
            for i in 0..hidden_size {
                out[dst_start + i] += weight * ex_out[src_start + i];
            }
            slot_cursor[ex] = slot + 1;
        }
    }

    out
}

// ──────────────────────────────────────────────────────────────────
// LCG weight initialisation (no rand crate)
// ──────────────────────────────────────────────────────────────────

/// Initialise `n` weights with small random values using a 64-bit LCG.
///
/// Values are in the range `(-scale, +scale)` where `scale = 1 / sqrt(n)`.
fn lcg_init_weights(n: usize, seed: u64) -> Vec<f32> {
    // Knuth's multiplicative LCG parameters
    const A: u64 = 6_364_136_223_846_793_005;
    const C: u64 = 1_442_695_040_888_963_407;
    let scale = if n > 0 { 1.0 / (n as f32).sqrt() } else { 1.0 };
    let mut state = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state = state.wrapping_mul(A).wrapping_add(C);
        // Map high 32 bits to [-1, 1] then scale
        let bits = (state >> 32) as u32;
        let f = (bits as f32 / u32::MAX as f32) * 2.0 - 1.0;
        out.push(f * scale);
    }
    out
}

// ──────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> MoeConfig {
        MoeConfig {
            num_experts: 4,
            top_k: 2,
            hidden_size: 16,
            capacity_factor: 1.25,
            z_loss_coeff: 1e-3,
            aux_loss_coeff: 1e-2,
        }
    }

    #[test]
    fn test_moe_config_default() {
        let cfg = MoeConfig::default();
        assert_eq!(cfg.num_experts, 8);
        assert_eq!(cfg.top_k, 2);
        assert_eq!(cfg.hidden_size, 256);
        assert!((cfg.capacity_factor - 1.25).abs() < 1e-6);
        assert!((cfg.z_loss_coeff - 1e-3).abs() < 1e-7);
        assert!((cfg.aux_loss_coeff - 1e-2).abs() < 1e-7);
    }

    #[test]
    fn test_top_k_router_route_output_shape() {
        let cfg = tiny_config();
        let router = TopKRouter::new(cfg.clone());
        let num_tokens = 6;
        let input = vec![0.1f32; num_tokens * cfg.hidden_size];
        let decision = router.route(&input, num_tokens);
        assert_eq!(decision.expert_indices.len(), num_tokens);
        assert_eq!(decision.expert_weights.len(), num_tokens);
        assert_eq!(decision.expert_load.len(), cfg.num_experts);
        assert_eq!(decision.overflow_mask.len(), num_tokens);
    }

    #[test]
    fn test_top_k_router_top_k_indices_valid() {
        let cfg = tiny_config();
        let router = TopKRouter::new(cfg.clone());
        let num_tokens = 8;
        let input: Vec<f32> = (0..num_tokens * cfg.hidden_size)
            .map(|i| (i as f32) * 0.01)
            .collect();
        let decision = router.route(&input, num_tokens);
        for token_indices in &decision.expert_indices {
            assert_eq!(token_indices.len(), cfg.top_k);
            for &idx in token_indices {
                assert!(idx < cfg.num_experts, "expert index {} out of range", idx);
            }
            // No duplicates within a token's top-k
            let mut seen = std::collections::HashSet::new();
            for &idx in token_indices {
                assert!(seen.insert(idx), "duplicate expert index {}", idx);
            }
        }
    }

    #[test]
    fn test_top_k_router_weights_sum_to_one_per_token() {
        let cfg = tiny_config();
        let router = TopKRouter::new(cfg.clone());
        let num_tokens = 5;
        let input = vec![0.5f32; num_tokens * cfg.hidden_size];
        let decision = router.route(&input, num_tokens);
        for weights in &decision.expert_weights {
            let sum: f32 = weights.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "weights do not sum to 1: {}", sum);
        }
    }

    #[test]
    fn test_routing_decision_load_balance() {
        let cfg = tiny_config();
        let router = TopKRouter::new(cfg.clone());
        let num_tokens = 16;
        let input: Vec<f32> = (0..num_tokens * cfg.hidden_size)
            .map(|i| ((i % 7) as f32) * 0.1)
            .collect();
        let decision = router.route(&input, num_tokens);
        let score = decision.load_balance_score();
        // Score must be in a reasonable range (can be negative for very unbalanced loads)
        assert!(score <= 1.0, "score {} should be <= 1.0", score);
        let total_load: usize = decision.expert_load.iter().sum();
        // Each token contributes top_k assignments (minus overflows)
        let overflowed = decision.overflow_mask.iter().filter(|&&b| b).count();
        let expected_max = num_tokens * cfg.top_k;
        assert!(
            total_load <= expected_max,
            "total load {} exceeds max {}",
            total_load,
            expected_max
        );
        let _ = overflowed;
    }

    #[test]
    fn test_expert_buffer_assign() {
        let mut buf = ExpertBuffer::new(4, 3);
        assert!(buf.try_assign(0, 0));
        assert!(buf.try_assign(1, 0));
        assert!(buf.try_assign(2, 0));
        assert_eq!(buf.counts[0], 3);
        assert_eq!(buf.tokens_for_expert(0), &[0usize, 1, 2]);
    }

    #[test]
    fn test_expert_buffer_overflow() {
        let mut buf = ExpertBuffer::new(2, 2);
        assert!(buf.try_assign(0, 0));
        assert!(buf.try_assign(1, 0));
        // Third token to same expert should fail
        assert!(!buf.try_assign(2, 0));
        assert_eq!(buf.overflow_count(), 1);
        assert_eq!(buf.counts[0], 2);
    }

    #[test]
    fn test_expert_buffer_utilization() {
        let mut buf = ExpertBuffer::new(2, 4);
        buf.try_assign(0, 0);
        buf.try_assign(1, 0);
        let util = buf.expert_utilization(0);
        assert!((util - 0.5).abs() < 1e-6, "utilization = {}", util);
        let zero_util = buf.expert_utilization(1);
        assert!((zero_util).abs() < 1e-6);
    }

    #[test]
    fn test_combine_expert_outputs_shape() {
        let num_tokens = 4;
        let hidden_size = 8;
        let num_experts = 2;
        let top_k = 1;

        // Build a simple routing decision: each token goes to expert 0
        let expert_indices: Vec<Vec<usize>> = (0..num_tokens).map(|_| vec![0]).collect();
        let expert_weights: Vec<Vec<f32>> = (0..num_tokens).map(|_| vec![1.0]).collect();
        let expert_load = vec![num_tokens, 0];
        let overflow_mask = vec![false; num_tokens];

        let routing = RoutingDecision {
            expert_indices,
            expert_weights,
            expert_load,
            overflow_mask,
            aux_loss: 0.0,
            z_loss: 0.0,
        };

        // Expert 0 outputs for all 4 tokens
        let ex0_out = vec![1.0f32; num_tokens * hidden_size];
        let ex1_out = vec![0.0f32; 0]; // no tokens assigned
        let expert_outputs = vec![ex0_out, ex1_out];

        let combined = combine_expert_outputs(&expert_outputs, &routing, num_tokens, hidden_size);
        assert_eq!(combined.len(), num_tokens * hidden_size);
        // Each output element should be 1.0 (weight=1.0, expert value=1.0)
        for &v in &combined {
            assert!((v - 1.0).abs() < 1e-6, "expected 1.0 got {}", v);
        }

        let _ = (num_experts, top_k);
    }

    #[test]
    fn test_z_loss_positive() {
        let logits = vec![1.0f32, 2.0, -1.0, 0.5, 3.0, -0.5];
        let z = TopKRouter::z_loss(&logits, 2, 3, 1e-3);
        assert!(z >= 0.0, "z_loss must be non-negative, got {}", z);
    }

    #[test]
    fn test_aux_loss_positive() {
        let expert_load = vec![5usize, 3, 4, 4];
        let num_tokens = 8usize;
        let num_experts = 4usize;
        // Uniform probs
        let expert_probs = vec![0.25f32; num_tokens * num_experts];
        let al = TopKRouter::aux_loss(&expert_load, &expert_probs, num_tokens, 1e-2);
        assert!(al >= 0.0, "aux_loss must be non-negative, got {}", al);
    }

    #[test]
    fn test_routing_decision_overflow_rate() {
        let mask = vec![true, false, true, false, false];
        let decision = RoutingDecision {
            expert_indices: vec![],
            expert_weights: vec![],
            expert_load: vec![],
            overflow_mask: mask,
            aux_loss: 0.0,
            z_loss: 0.0,
        };
        let rate = decision.overflow_rate();
        assert!((rate - 0.4).abs() < 1e-6, "expected 0.4 got {}", rate);
    }
}
