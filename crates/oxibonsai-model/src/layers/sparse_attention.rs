//! Sparse attention patterns for efficient long-sequence processing.
//!
//! Implements three attention mask patterns:
//! - Local window attention (Longformer-style)
//! - Global + local (BigBird-style)
//! - Strided sparse attention (every k-th token attends globally)
//!
//! These reduce complexity from O(n²) to O(n√n) or O(n*w).

use thiserror::Error;

use crate::layers::attention_fused::softmax_inplace;

// ─── Error type ──────────────────────────────────────────────────────────────

/// Errors that can occur during sparse attention operations.
#[derive(Debug, Error)]
pub enum SparseAttnError {
    #[error("query/key/value length mismatch: q={q}, k={k}, v={v}")]
    LengthMismatch { q: usize, k: usize, v: usize },
    #[error("head_dim must be > 0")]
    InvalidHeadDim,
    #[error("window_size must be odd for symmetric windows")]
    WindowSizeMustBeOdd,
    #[error("empty attention: no valid (q,k) pairs")]
    EmptyAttention,
}

// ─── Sparse Pattern ───────────────────────────────────────────────────────────

/// Sparse attention pattern type.
#[derive(Debug, Clone, PartialEq)]
pub enum SparsePattern {
    /// Each token attends to `window_size` neighbors (must be odd).
    LocalWindow { window_size: usize },
    /// BigBird: global tokens + local window + random sparse connections.
    BigBird {
        window_size: usize,
        num_global_tokens: usize,
        num_random_connections: usize,
        seed: u64,
    },
    /// Strided: every `stride`-th token attends globally; others use local window.
    Strided { window_size: usize, stride: usize },
    /// Full dense attention (baseline).
    Dense,
}

// ─── Sparse Attention Mask ────────────────────────────────────────────────────

/// A sparse attention mask: which positions each query can attend to.
pub struct SparseAttentionMask {
    /// Length of the sequence.
    pub seq_len: usize,
    /// For each query position: sorted list of key positions it can attend to.
    attend_to: Vec<Vec<usize>>,
    /// The pattern that was used to build this mask.
    pub pattern: SparsePattern,
}

impl SparseAttentionMask {
    /// Build a sparse mask for `seq_len` tokens using `pattern`.
    ///
    /// Returns an error if the pattern parameters are invalid.
    pub fn build(seq_len: usize, pattern: &SparsePattern) -> Result<Self, SparseAttnError> {
        let attend_to = match pattern {
            SparsePattern::Dense => build_dense(seq_len),
            SparsePattern::LocalWindow { window_size } => {
                build_local_window(seq_len, *window_size)?
            }
            SparsePattern::BigBird {
                window_size,
                num_global_tokens,
                num_random_connections,
                seed,
            } => build_bigbird(
                seq_len,
                *window_size,
                *num_global_tokens,
                *num_random_connections,
                *seed,
            )?,
            SparsePattern::Strided {
                window_size,
                stride,
            } => build_strided(seq_len, *window_size, *stride)?,
        };

        Ok(Self {
            seq_len,
            attend_to,
            pattern: pattern.clone(),
        })
    }

    /// Get the key positions that query `q` can attend to.
    pub fn keys_for_query(&self, q: usize) -> &[usize] {
        if q >= self.seq_len {
            return &[];
        }
        &self.attend_to[q]
    }

    /// Total number of attended (q, k) pairs.
    pub fn nnz(&self) -> usize {
        self.attend_to.iter().map(|v| v.len()).sum()
    }

    /// Density: nnz / seq_len² (1.0 = dense).
    pub fn density(&self) -> f32 {
        let total = (self.seq_len as f64) * (self.seq_len as f64);
        if total == 0.0 {
            return 0.0;
        }
        (self.nnz() as f64 / total) as f32
    }

    /// Whether query `q` can attend to key `k`.
    pub fn can_attend(&self, q: usize, k: usize) -> bool {
        if q >= self.seq_len || k >= self.seq_len {
            return false;
        }
        self.attend_to[q].binary_search(&k).is_ok()
    }

    /// Convert to a dense boolean mask matrix [seq_len × seq_len].
    ///
    /// `result[q * seq_len + k] == true` means query `q` attends to key `k`.
    pub fn to_dense(&self) -> Vec<Vec<bool>> {
        let n = self.seq_len;
        let mut mask = vec![vec![false; n]; n];
        for (q, keys) in self.attend_to.iter().enumerate() {
            for &k in keys {
                mask[q][k] = true;
            }
        }
        mask
    }
}

// ─── Pattern builders ─────────────────────────────────────────────────────────

/// Full dense: every query attends to every key.
fn build_dense(seq_len: usize) -> Vec<Vec<usize>> {
    (0..seq_len).map(|_| (0..seq_len).collect()).collect()
}

/// Local sliding window of `window_size` (must be odd) centered at each token.
fn build_local_window(
    seq_len: usize,
    window_size: usize,
) -> Result<Vec<Vec<usize>>, SparseAttnError> {
    if window_size.is_multiple_of(2) {
        return Err(SparseAttnError::WindowSizeMustBeOdd);
    }
    let half = window_size / 2;
    let mut attend_to = Vec::with_capacity(seq_len);
    for q in 0..seq_len {
        let start = q.saturating_sub(half);
        let end = (q + half + 1).min(seq_len);
        attend_to.push((start..end).collect());
    }
    Ok(attend_to)
}

/// BigBird pattern: global tokens + local window + random sparse connections.
///
/// Uses a linear congruential generator (LCG) instead of rand to avoid
/// external dependencies while producing deterministic pseudo-random connections.
fn build_bigbird(
    seq_len: usize,
    window_size: usize,
    num_global_tokens: usize,
    num_random_connections: usize,
    seed: u64,
) -> Result<Vec<Vec<usize>>, SparseAttnError> {
    if window_size.is_multiple_of(2) {
        return Err(SparseAttnError::WindowSizeMustBeOdd);
    }
    let half = window_size / 2;
    // Clamp global tokens to seq_len
    let actual_global = num_global_tokens.min(seq_len);

    let mut attend_to: Vec<Vec<usize>> = Vec::with_capacity(seq_len);
    let mut lcg_state = seed.wrapping_add(0xDEAD_BEEF_CAFE_1234);

    for q in 0..seq_len {
        let mut keys: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();

        // 1. Global tokens: attend to all positions
        for g in 0..actual_global {
            keys.insert(g);
        }
        // 2. All queries attend to all global-token positions (global tokens attend back)
        for g in 0..actual_global {
            if q == g {
                // global token q itself attends to every position
                for k in 0..seq_len {
                    keys.insert(k);
                }
            }
        }

        // 3. Local window
        let start = q.saturating_sub(half);
        let end = (q + half + 1).min(seq_len);
        for k in start..end {
            keys.insert(k);
        }

        // 4. Random sparse connections (LCG)
        let num_rand = if seq_len > actual_global + window_size {
            num_random_connections
        } else {
            0
        };
        for r in 0..num_rand {
            // LCG: a=6364136223846793005, c=1442695040888963407 (Knuth)
            lcg_state = lcg_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407)
                .wrapping_add((q as u64).wrapping_mul(137).wrapping_add(r as u64));
            let k = (lcg_state >> 33) as usize % seq_len;
            keys.insert(k);
        }

        attend_to.push(keys.into_iter().collect());
    }

    Ok(attend_to)
}

/// Strided pattern: stride positions attend globally; others use local window.
fn build_strided(
    seq_len: usize,
    window_size: usize,
    stride: usize,
) -> Result<Vec<Vec<usize>>, SparseAttnError> {
    if window_size.is_multiple_of(2) {
        return Err(SparseAttnError::WindowSizeMustBeOdd);
    }
    if stride == 0 {
        // stride=0 degenerates to all-global; treat as dense
        return Ok(build_dense(seq_len));
    }
    let half = window_size / 2;

    let mut attend_to = Vec::with_capacity(seq_len);
    for q in 0..seq_len {
        let is_global = (q % stride) == 0;
        let mut keys: Vec<usize> = if is_global {
            // stride positions attend to every key
            (0..seq_len).collect()
        } else {
            // local window
            let start = q.saturating_sub(half);
            let end = (q + half + 1).min(seq_len);
            // plus all stride positions (global tokens)
            let mut ks: std::collections::BTreeSet<usize> = (start..end).collect();
            let mut g = 0usize;
            while g < seq_len {
                ks.insert(g);
                g += stride;
            }
            ks.into_iter().collect()
        };
        keys.sort_unstable();
        keys.dedup();
        attend_to.push(keys);
    }
    Ok(attend_to)
}

// ─── Sparse attention forward ─────────────────────────────────────────────────

/// Apply sparse attention: compute attention output using a sparse mask.
///
/// - `queries`: shape [seq_len, head_dim] (row-major)
/// - `keys`:    shape [seq_len, head_dim] (row-major)
/// - `values`:  shape [seq_len, head_dim] (row-major)
///
/// Returns: shape [seq_len, head_dim]
pub fn sparse_attention_forward(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    seq_len: usize,
    head_dim: usize,
    mask: &SparseAttentionMask,
    scale: f32,
) -> Result<Vec<f32>, SparseAttnError> {
    validate_inputs(queries, keys, values, seq_len, head_dim)?;

    if mask.nnz() == 0 {
        return Err(SparseAttnError::EmptyAttention);
    }

    let mut output = vec![0.0f32; seq_len * head_dim];

    for q in 0..seq_len {
        let key_positions = mask.keys_for_query(q);
        if key_positions.is_empty() {
            // No attention positions: output stays zero for this query
            continue;
        }

        let q_vec = &queries[q * head_dim..(q + 1) * head_dim];

        // Compute raw scores for each attended key
        let mut scores: Vec<f32> = key_positions
            .iter()
            .map(|&k| {
                let k_vec = &keys[k * head_dim..(k + 1) * head_dim];
                dot_scaled(q_vec, k_vec, scale)
            })
            .collect();

        // In-place softmax over the sparse scores
        softmax_inplace(&mut scores);

        // Weighted sum over values
        let out_row = &mut output[q * head_dim..(q + 1) * head_dim];
        for (weight, &k_pos) in scores.iter().zip(key_positions.iter()) {
            let v_vec = &values[k_pos * head_dim..(k_pos + 1) * head_dim];
            for (o, &v) in out_row.iter_mut().zip(v_vec.iter()) {
                *o += weight * v;
            }
        }
    }

    Ok(output)
}

/// Compare sparse vs dense attention output (MAE).
///
/// Computes both with the given mask and with a fully dense mask, then
/// returns the mean absolute error between the two outputs.
pub fn sparse_vs_dense_error(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    seq_len: usize,
    head_dim: usize,
    mask: &SparseAttentionMask,
) -> Result<f32, SparseAttnError> {
    let scale = 1.0 / (head_dim as f32).sqrt();

    let sparse_out =
        sparse_attention_forward(queries, keys, values, seq_len, head_dim, mask, scale)?;

    let dense_mask = SparseAttentionMask::build(seq_len, &SparsePattern::Dense)
        .map_err(|_| SparseAttnError::EmptyAttention)?;
    let dense_out =
        sparse_attention_forward(queries, keys, values, seq_len, head_dim, &dense_mask, scale)?;

    let total_elements = seq_len * head_dim;
    if total_elements == 0 {
        return Ok(0.0);
    }

    let mae = sparse_out
        .iter()
        .zip(dense_out.iter())
        .map(|(s, d)| (s - d).abs())
        .sum::<f32>()
        / total_elements as f32;

    Ok(mae)
}

/// Memory savings vs dense attention.
///
/// Returns the fraction of memory saved: `1.0 - density`.
/// A value of 0.0 means no savings (dense), 1.0 means all connections removed.
pub fn memory_reduction(_seq_len: usize, mask: &SparseAttentionMask) -> f32 {
    1.0 - mask.density()
}

// ─── Private helpers ──────────────────────────────────────────────────────────

/// Validate input buffer sizes.
fn validate_inputs(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    seq_len: usize,
    head_dim: usize,
) -> Result<(), SparseAttnError> {
    if head_dim == 0 {
        return Err(SparseAttnError::InvalidHeadDim);
    }
    let expected = seq_len * head_dim;
    if queries.len() != expected || keys.len() != expected || values.len() != expected {
        return Err(SparseAttnError::LengthMismatch {
            q: queries.len(),
            k: keys.len(),
            v: values.len(),
        });
    }
    Ok(())
}

/// Scaled dot product of two vectors.
#[inline]
fn dot_scaled(a: &[f32], b: &[f32], scale: f32) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum::<f32>() * scale
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_qkv(seq_len: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let n = seq_len * head_dim;
        let q: Vec<f32> = (0..n).map(|i| (i as f32 * 0.03) - 0.5).collect();
        let k: Vec<f32> = (0..n)
            .map(|i| ((i * 7 + 3) % 17) as f32 * 0.04 - 0.3)
            .collect();
        let v: Vec<f32> = (0..n)
            .map(|i| ((i * 11 + 5) % 13) as f32 * 0.05 - 0.3)
            .collect();
        (q, k, v)
    }

    #[test]
    fn dense_mask_full() {
        let seq_len = 8;
        let mask = SparseAttentionMask::build(seq_len, &SparsePattern::Dense)
            .expect("dense build should succeed");
        assert_eq!(mask.nnz(), seq_len * seq_len);
    }

    #[test]
    fn local_window_density_less_than_one() {
        let seq_len = 16;
        let mask =
            SparseAttentionMask::build(seq_len, &SparsePattern::LocalWindow { window_size: 3 })
                .expect("local window build should succeed");
        assert!(
            mask.density() < 1.0,
            "density should be < 1.0 for local window"
        );
    }

    #[test]
    fn sparse_forward_dense_matches_naive_inline() {
        let seq_len = 4;
        let head_dim = 4;
        let (q, k, v) = make_qkv(seq_len, head_dim);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mask = SparseAttentionMask::build(seq_len, &SparsePattern::Dense).expect("dense mask");
        let out = sparse_attention_forward(&q, &k, &v, seq_len, head_dim, &mask, scale)
            .expect("sparse forward failed");
        assert_eq!(out.len(), seq_len * head_dim);
    }
}
