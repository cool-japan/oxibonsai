//! Qwen3 rotary position embedding (HF "half-split" convention).
//!
//! This is **not** the interleaved RoPE used by the DiT ([`crate::math`]). The
//! Qwen3 text encoder uses the HuggingFace `rotate_half` formulation:
//!
//! ```text
//! inv_freq[i] = base^(-(2 i) / head_dim)            for i in 0..head_dim/2
//! freqs[t, i] = position[t] * inv_freq[i]           [seq, head_dim/2]
//! emb[t]      = concat(freqs[t], freqs[t])          [seq, head_dim]
//! cos = cos(emb), sin = sin(emb)                    [seq, head_dim]
//!
//! rotate_half(x) = concat(-x[half:], x[:half])
//! x' = x * cos + rotate_half(x) * sin
//! ```
//!
//! (Matches `Qwen3TextRotaryEmbedding` + `Qwen3VLAttention._apply_rotary_pos_emb`
//! / `_rotate_half` in the reference mflux source.)

/// Precomputed Qwen3 RoPE `cos`/`sin` tables, each `[seq, head_dim]`.
#[derive(Debug, Clone)]
pub struct Qwen3Rope {
    /// `[seq, head_dim]` cos table.
    pub cos: Vec<f32>,
    /// `[seq, head_dim]` sin table.
    pub sin: Vec<f32>,
    /// Sequence length.
    pub seq: usize,
    /// Per-head dimension (even).
    pub head_dim: usize,
}

impl Qwen3Rope {
    /// Build the tables for positions `0..seq` with the given `head_dim` and
    /// `base` (`theta`). The first `head_dim/2` columns repeat in the second
    /// half (`emb = concat(freqs, freqs)`).
    pub fn new(seq: usize, head_dim: usize, base: f32) -> Self {
        let half = head_dim / 2;
        // inv_freq[i] = base^(-(2 i) / head_dim)  (computed in f64 for accuracy,
        // matching MLX's float32 arange/divide then power).
        let mut inv_freq = vec![0.0f32; half];
        for (i, f) in inv_freq.iter_mut().enumerate() {
            let exponent = (2 * i) as f64 / head_dim as f64;
            *f = (base as f64).powf(-exponent) as f32;
        }
        let mut cos = vec![0.0f32; seq * head_dim];
        let mut sin = vec![0.0f32; seq * head_dim];
        for t in 0..seq {
            let pos = t as f32;
            let crow = &mut cos[t * head_dim..(t + 1) * head_dim];
            let srow = &mut sin[t * head_dim..(t + 1) * head_dim];
            for i in 0..half {
                let angle = pos * inv_freq[i];
                let (s, c) = angle.sin_cos();
                // emb = concat(freqs, freqs): columns i and half+i share `angle`.
                crow[i] = c;
                crow[half + i] = c;
                srow[i] = s;
                srow[half + i] = s;
            }
        }
        Self {
            cos,
            sin,
            seq,
            head_dim,
        }
    }

    /// Apply RoPE in-place to a `[num_heads, seq, head_dim]` head-major buffer
    /// using the half-split `rotate_half` rule (broadcast over heads).
    ///
    /// `x'[d]      = x[d]      * cos[d]      - x[half+d] * sin[d]`        (d < half)
    /// `x'[half+d] = x[half+d] * cos[half+d] + x[d]      * sin[half+d]`
    ///
    /// Since `cos[d] == cos[half+d]` and `sin[d] == sin[half+d]`, both halves use
    /// the same per-pair `(c, s)`.
    pub fn apply(&self, x: &mut [f32], num_heads: usize, seq: usize) {
        debug_assert_eq!(self.seq, seq);
        debug_assert_eq!(x.len(), num_heads * seq * self.head_dim);
        let head_dim = self.head_dim;
        let half = head_dim / 2;
        let (cos, sin) = (&self.cos, &self.sin);
        // Rotate one head's `[seq, head_dim]` block in place.
        let rotate_head = |block: &mut [f32]| {
            for t in 0..seq {
                let crow = &cos[t * head_dim..(t + 1) * head_dim];
                let srow = &sin[t * head_dim..(t + 1) * head_dim];
                let row = &mut block[t * head_dim..(t + 1) * head_dim];
                for d in 0..half {
                    let lo = row[d];
                    let hi = row[half + d];
                    let c = crow[d];
                    let s = srow[d];
                    // rotate_half(x) = [-hi, lo]
                    row[d] = lo * c - hi * s;
                    row[half + d] = hi * c + lo * s;
                }
            }
        };
        // Heads are independent → parallel across CPUs (bit-identical to serial).
        let threads = std::thread::available_parallelism()
            .map(|t| t.get())
            .unwrap_or(1)
            .min(num_heads.max(1));
        let head_len = seq * head_dim;
        if threads <= 1 || num_heads < 4 {
            for block in x.chunks_mut(head_len) {
                rotate_head(block);
            }
            return;
        }
        let per = num_heads.div_ceil(threads);
        let rotate_ref = &rotate_head;
        std::thread::scope(|scope| {
            for blocks in x.chunks_mut(per * head_len) {
                scope.spawn(move || {
                    for block in blocks.chunks_mut(head_len) {
                        rotate_ref(block);
                    }
                });
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pos_zero_is_identity() {
        let rope = Qwen3Rope::new(1, 8, 1_000_000.0);
        assert!(rope.cos.iter().all(|&c| (c - 1.0).abs() < 1e-9));
        assert!(rope.sin.iter().all(|&s| s.abs() < 1e-9));
        let mut x: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let orig = x.clone();
        rope.apply(&mut x, 1, 1);
        assert_eq!(x, orig);
    }

    #[test]
    fn rotate_half_matches_reference() {
        // head_dim=4, half=2. For position t=1, base=10.
        // inv_freq = [10^0, 10^(-0.5)] = [1, 0.31623]
        let base = 10.0f32;
        let seq = 2usize;
        let head_dim = 4usize;
        let rope = Qwen3Rope::new(seq, head_dim, base);
        // 1 head, 2 tokens, head_dim 4 → 8 elements. Token 0 = [0,0,0,0] (pos 0,
        // identity), token 1 = [1,2,3,4].
        let mut x = vec![0.0f32, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0];
        let t = 1usize;
        let inv0 = 1.0f32;
        let inv1 = base.powf(-0.5);
        let (s0, c0) = (t as f32 * inv0).sin_cos();
        let (s1, c1) = (t as f32 * inv1).sin_cos();
        // x = [a,b,c,d] -> rotate_half = [-c,-d,a,b]; x' = x*cos + rh*sin
        let e0 = 1.0 * c0 - 3.0 * s0;
        let e1 = 2.0 * c1 - 4.0 * s1;
        let e2 = 3.0 * c0 + 1.0 * s0;
        let e3 = 4.0 * c1 + 2.0 * s1;
        rope.apply(&mut x, 1, seq);
        let row = &x[t * head_dim..t * head_dim + head_dim];
        assert!((row[0] - e0).abs() < 1e-5, "{} vs {e0}", row[0]);
        assert!((row[1] - e1).abs() < 1e-5, "{} vs {e1}", row[1]);
        assert!((row[2] - e2).abs() < 1e-5, "{} vs {e2}", row[2]);
        assert!((row[3] - e3).abs() < 1e-5, "{} vs {e3}", row[3]);
    }
}
