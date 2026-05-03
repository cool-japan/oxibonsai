//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(test)]
use crate::block::types::{LayerStats, TransformerBlock};
use crate::error::{ModelError, ModelResult};
use crate::kv_cache::KvCache;
use crate::layers::attention_fused::fused_attention_head_contiguous;
#[cfg(test)]
use crate::layers::linear::Linear1Bit;
#[cfg(test)]
use crate::layers::rms_norm::RmsNorm;
#[cfg(test)]
use crate::layers::rope::RopeTable;
#[cfg(test)]
use crate::layers::sliding_window::SlidingWindowConfig;
use rayon::prelude::*;

/// Convert a BlockQ1_0G128 slice to raw bytes (zero-copy).
///
/// # Safety
/// `BlockQ1_0G128` is `#[repr(C)]` with a well-defined 18-byte layout.
#[cfg(any(
    feature = "metal",
    all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    )
))]
pub(crate) fn blocks_as_bytes(blocks: &[oxibonsai_core::BlockQ1_0G128]) -> &[u8] {
    let ptr = blocks.as_ptr() as *const u8;
    let len = std::mem::size_of_val(blocks);
    unsafe { std::slice::from_raw_parts(ptr, len) }
}
/// Reinterpret a slice of ternary blocks as raw bytes (zero-copy).
///
/// Used by the Metal and CUDA ternary full-forward paths to feed AoS bytes
/// into the GPU weight cache — gated on GPU features to avoid dead-code
/// warnings on CPU-only builds.
///
/// # Safety
/// `BlockTQ2_0_g128` is `#[repr(C)]` with a 34-byte layout `(qs: [u8;32], d: f16)`,
/// so the cast is valid.
#[cfg(any(
    all(feature = "metal", target_os = "macos"),
    all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    )
))]
pub(crate) fn blocks_as_bytes_ternary(blocks: &[oxibonsai_core::BlockTQ2_0_g128]) -> &[u8] {
    let ptr = blocks.as_ptr() as *const u8;
    let len = std::mem::size_of_val(blocks);
    unsafe { std::slice::from_raw_parts(ptr, len) }
}
/// Minimum number of Q heads required to engage the Rayon parallel path.
///
/// For models with fewer than this many Q heads the per-head work is too
/// small to amortise Rayon's thread-pool overhead, so we fall back to the
/// sequential loop.
pub(super) const PAR_HEAD_MIN_HEADS: usize = 8;
/// Run GQA attention for all Q heads, writing results into `attn_out`.
///
/// Dispatches to a parallel Rayon loop when `num_q_heads >= PAR_HEAD_MIN_HEADS`,
/// otherwise runs sequentially to avoid thread-pool overhead on small models.
///
/// # Arguments
/// - `q_rope`: All Q-head vectors concatenated `[num_q_heads * head_dim]`.
/// - `attn_out`: Output buffer `[num_q_heads * head_dim]`.
/// - `kv_cache`: KV cache (read-only for this call).
/// - `layer_idx`: Layer index for KV-cache lookup.
/// - `num_q_heads`: Number of Q heads.
/// - `heads_per_group`: `num_q_heads / num_kv_heads` (GQA ratio).
/// - `head_dim`: Dimension per head.
/// - `seq_len`: Number of KV positions to attend to.
#[allow(clippy::too_many_arguments)]
pub(super) fn compute_gqa_attention(
    q_rope: &[f32],
    attn_out: &mut [f32],
    kv_cache: &KvCache,
    layer_idx: usize,
    num_q_heads: usize,
    heads_per_group: usize,
    head_dim: usize,
    seq_len: usize,
) -> ModelResult<()> {
    if num_q_heads >= PAR_HEAD_MIN_HEADS {
        attn_out.par_chunks_mut(head_dim).enumerate().try_for_each(
            |(q_head, out_slice)| -> ModelResult<()> {
                let kv_head = q_head / heads_per_group;
                let q_start = q_head * head_dim;
                let keys = kv_cache.keys_for(layer_idx, kv_head, seq_len);
                let values = kv_cache.values_for(layer_idx, kv_head, seq_len);
                fused_attention_head_contiguous(
                    &q_rope[q_start..q_start + head_dim],
                    keys,
                    values,
                    out_slice,
                    seq_len,
                    head_dim,
                )
                .map_err(|e| ModelError::Internal(format!("parallel head {q_head} attention: {e}")))
            },
        )
    } else {
        for q_head in 0..num_q_heads {
            let kv_head = q_head / heads_per_group;
            let q_start = q_head * head_dim;
            let keys = kv_cache.keys_for(layer_idx, kv_head, seq_len);
            let values = kv_cache.values_for(layer_idx, kv_head, seq_len);
            fused_attention_head_contiguous(
                &q_rope[q_start..q_start + head_dim],
                keys,
                values,
                &mut attn_out[q_start..q_start + head_dim],
                seq_len,
                head_dim,
            )?;
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;
    use oxibonsai_core::tensor::BlockQ1_0G128;
    fn make_blocks(n: usize, scale: f32, pattern: u8) -> Vec<BlockQ1_0G128> {
        (0..n)
            .map(|_| BlockQ1_0G128 {
                d: f16::from_f32(scale),
                qs: [pattern; 16],
            })
            .collect()
    }
    /// Create a minimal test block with the given dimensions.
    #[allow(clippy::too_many_arguments)]
    fn make_test_block<'a>(
        h: usize,
        hd: usize,
        nq: usize,
        nkv: usize,
        inter: usize,
        kernel: std::sync::Arc<oxibonsai_kernels::KernelDispatcher>,
        q_blocks: &'a [BlockQ1_0G128],
        k_blocks: &'a [BlockQ1_0G128],
        v_blocks: &'a [BlockQ1_0G128],
        o_blocks: &'a [BlockQ1_0G128],
        gate_blocks: &'a [BlockQ1_0G128],
        up_blocks: &'a [BlockQ1_0G128],
        down_blocks: &'a [BlockQ1_0G128],
    ) -> TransformerBlock<'a> {
        TransformerBlock::new(
            0,
            RmsNorm::new(vec![1.0; h], 1e-6),
            Linear1Bit::new(q_blocks, nq * hd, h, kernel.clone())
                .expect("q")
                .into(),
            Linear1Bit::new(k_blocks, nkv * hd, h, kernel.clone())
                .expect("k")
                .into(),
            Linear1Bit::new(v_blocks, nkv * hd, h, kernel.clone())
                .expect("v")
                .into(),
            Linear1Bit::new(o_blocks, h, nq * hd, kernel.clone())
                .expect("o")
                .into(),
            RmsNorm::new(vec![1.0; hd], 1e-6),
            RmsNorm::new(vec![1.0; hd], 1e-6),
            RmsNorm::new(vec![1.0; h], 1e-6),
            Linear1Bit::new(gate_blocks, inter, h, kernel.clone())
                .expect("gate")
                .into(),
            Linear1Bit::new(up_blocks, inter, h, kernel.clone())
                .expect("up")
                .into(),
            Linear1Bit::new(down_blocks, h, inter, kernel)
                .expect("down")
                .into(),
            nq,
            nkv,
            hd,
            h,
        )
    }
    #[test]
    fn transformer_block_smoke_test() {
        let (h, hd, nq, nkv, inter) = (128, 64, 2, 1, 256);
        let bpr = h / 128;
        let q_b = make_blocks(nq * hd * bpr, 0.01, 0xFF);
        let k_b = make_blocks(nkv * hd * bpr, 0.01, 0xFF);
        let v_b = make_blocks(nkv * hd * bpr, 0.01, 0xFF);
        let o_b = make_blocks(h * bpr, 0.01, 0xFF);
        let g_b = make_blocks(inter * bpr, 0.01, 0xFF);
        let u_b = make_blocks(inter * bpr, 0.01, 0xFF);
        let d_b = make_blocks(h * (inter / 128), 0.01, 0xFF);
        let kernel = std::sync::Arc::new(oxibonsai_kernels::KernelDispatcher::auto_detect());
        let block = make_test_block(
            h,
            hd,
            nq,
            nkv,
            inter,
            kernel.clone(),
            &q_b,
            &k_b,
            &v_b,
            &o_b,
            &g_b,
            &u_b,
            &d_b,
        );
        let rope = RopeTable::new(hd, 16, 10000.0);
        let mut kv_cache = KvCache::new(1, nkv, hd, 16);
        let mut hidden: Vec<f32> = (0..h).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let original = hidden.clone();
        block
            .forward(&mut hidden, 0, &mut kv_cache, &rope, kernel.as_ref())
            .expect("block forward should succeed");
        let max_diff = hidden
            .iter()
            .zip(original.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-6,
            "forward should modify hidden state, max_diff={max_diff}"
        );
    }
    #[test]
    fn forward_with_stats_returns_timing() {
        let (h, hd, nq, nkv, inter) = (128, 64, 2, 1, 256);
        let bpr = h / 128;
        let q_b = make_blocks(nq * hd * bpr, 0.01, 0xFF);
        let k_b = make_blocks(nkv * hd * bpr, 0.01, 0xFF);
        let v_b = make_blocks(nkv * hd * bpr, 0.01, 0xFF);
        let o_b = make_blocks(h * bpr, 0.01, 0xFF);
        let g_b = make_blocks(inter * bpr, 0.01, 0xFF);
        let u_b = make_blocks(inter * bpr, 0.01, 0xFF);
        let d_b = make_blocks(h * (inter / 128), 0.01, 0xFF);
        let kernel = std::sync::Arc::new(oxibonsai_kernels::KernelDispatcher::auto_detect());
        let block = make_test_block(
            h,
            hd,
            nq,
            nkv,
            inter,
            kernel.clone(),
            &q_b,
            &k_b,
            &v_b,
            &o_b,
            &g_b,
            &u_b,
            &d_b,
        );
        let rope = RopeTable::new(hd, 16, 10000.0);
        let mut kv_cache = KvCache::new(1, nkv, hd, 16);
        let mut hidden: Vec<f32> = (0..h).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let stats = block
            .forward_with_stats(&mut hidden, 0, &mut kv_cache, &rope, kernel.as_ref())
            .expect("forward_with_stats should succeed");
        assert_eq!(stats.layer_idx, 0);
        assert!(stats.total_us >= stats.projection_us.min(stats.attention_us));
    }
    #[test]
    fn forward_with_sliding_window_smoke() {
        let (h, hd, nq, nkv, inter) = (128, 64, 2, 1, 256);
        let bpr = h / 128;
        let q_b = make_blocks(nq * hd * bpr, 0.01, 0xFF);
        let k_b = make_blocks(nkv * hd * bpr, 0.01, 0xFF);
        let v_b = make_blocks(nkv * hd * bpr, 0.01, 0xFF);
        let o_b = make_blocks(h * bpr, 0.01, 0xFF);
        let g_b = make_blocks(inter * bpr, 0.01, 0xFF);
        let u_b = make_blocks(inter * bpr, 0.01, 0xFF);
        let d_b = make_blocks(h * (inter / 128), 0.01, 0xFF);
        let kernel = std::sync::Arc::new(oxibonsai_kernels::KernelDispatcher::auto_detect());
        let block = make_test_block(
            h,
            hd,
            nq,
            nkv,
            inter,
            kernel.clone(),
            &q_b,
            &k_b,
            &v_b,
            &o_b,
            &g_b,
            &u_b,
            &d_b,
        );
        let rope = RopeTable::new(hd, 16, 10000.0);
        let mut kv_cache = KvCache::new(1, nkv, hd, 16);
        let sw_config = SlidingWindowConfig::new(8, 2);
        let mut hidden: Vec<f32> = (0..h).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let original = hidden.clone();
        block
            .forward_with_sliding_window(
                &mut hidden,
                0,
                &mut kv_cache,
                &rope,
                kernel.as_ref(),
                Some(&sw_config),
            )
            .expect("forward_with_sliding_window should succeed");
        let max_diff = hidden
            .iter()
            .zip(original.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff > 1e-6);
    }
    /// P11.3: Smoke test the parallel attention path (nq=8 >= PAR_HEAD_MIN_HEADS).
    #[test]
    fn parallel_attention_smoke() {
        let h = 128;
        let hd = 16;
        let nq = 8;
        let nkv = 2;
        let inter = 256;
        let bpr = h / 128;
        let q_b = make_blocks(nq * hd * bpr, 0.01, 0xFF);
        let k_b = make_blocks(nkv * hd * bpr, 0.01, 0xFF);
        let v_b = make_blocks(nkv * hd * bpr, 0.01, 0xFF);
        let o_b = make_blocks(h * bpr, 0.01, 0xFF);
        let g_b = make_blocks(inter * bpr, 0.01, 0xFF);
        let u_b = make_blocks(inter * bpr, 0.01, 0xFF);
        let d_b = make_blocks(h * (inter / 128), 0.01, 0xFF);
        let kernel = std::sync::Arc::new(oxibonsai_kernels::KernelDispatcher::auto_detect());
        let block = make_test_block(
            h,
            hd,
            nq,
            nkv,
            inter,
            kernel.clone(),
            &q_b,
            &k_b,
            &v_b,
            &o_b,
            &g_b,
            &u_b,
            &d_b,
        );
        let rope = RopeTable::new(hd, 32, 10000.0);
        let mut kv_cache = KvCache::new(1, nkv, hd, 32);
        let mut hidden: Vec<f32> = (0..h).map(|i| (i as f32 + 1.0) * 0.005).collect();
        let original = hidden.clone();
        block
            .forward(&mut hidden, 0, &mut kv_cache, &rope, kernel.as_ref())
            .expect("parallel attention forward should succeed");
        let max_diff = hidden
            .iter()
            .zip(original.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-6,
            "parallel forward (nq={nq} >= PAR_HEAD_MIN_HEADS={PAR_HEAD_MIN_HEADS}) should modify hidden, max_diff={max_diff}"
        );
    }
    #[test]
    fn layer_stats_fractions() {
        let mut stats = LayerStats::new(0);
        stats.total_us = 100;
        stats.attention_us = 60;
        stats.ffn_us = 30;
        assert!((stats.attention_fraction() - 0.6).abs() < 1e-10);
        assert!((stats.ffn_fraction() - 0.3).abs() < 1e-10);
    }
    #[test]
    fn layer_stats_zero_total() {
        let stats = LayerStats::new(5);
        assert!((stats.attention_fraction() - 0.0).abs() < 1e-10);
        assert!((stats.ffn_fraction() - 0.0).abs() < 1e-10);
    }
}
