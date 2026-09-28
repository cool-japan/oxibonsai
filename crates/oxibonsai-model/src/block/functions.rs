//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(test)]
use crate::block::types::{LayerStats, TransformerBlock};
use crate::error::{ModelError, ModelResult};
use crate::kv_cache::KvCache;
#[cfg(test)]
use crate::layers::linear::Linear1Bit;
#[cfg(test)]
use crate::layers::linear::LinearTernary;
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
/// See [`gqa_attention`] for how the cache is read.
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
    gqa_attention(
        q_rope,
        attn_out,
        kv_cache,
        layer_idx,
        num_q_heads,
        heads_per_group,
        head_dim,
        seq_len,
        num_q_heads >= PAR_HEAD_MIN_HEADS,
    )
}

/// GQA attention of every query head of one position against `layer_idx`'s
/// cached history `0..seq_len`, one [`KvCache::attend_group`] call per KV
/// head (optionally in parallel across KV heads).
///
/// The cache is read in place whatever its element type (B2-11-FIX): an
/// `f32` cache runs each query head through `fused_attention_head_contiguous`
/// over zero-copy slices — the exact bits this function has always
/// produced — and an `f16` cache (the host default) widens each history row
/// once per KV head on the stack. There is no history copy on either path;
/// the `Cow`/`keys_for_owned` detour the sparse cache used to take (one
/// `seq_len × head_dim` allocation per head per token) is gone.
///
/// Shared by the per-block decode path and the batched CPU prefill
/// (`model/types/prefill_cpu.rs`), so the two cannot drift.
///
/// # Errors
///
/// [`ModelError::ShapeInvariant`] for a zero GQA group or head width, or a
/// `q_rope`/`attn_out` shorter than `num_q_heads * head_dim`; anything
/// [`KvCache::attend_group`] returns.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gqa_attention(
    q_rope: &[f32],
    attn_out: &mut [f32],
    kv_cache: &KvCache,
    layer_idx: usize,
    num_q_heads: usize,
    heads_per_group: usize,
    head_dim: usize,
    seq_len: usize,
    parallel: bool,
) -> ModelResult<()> {
    let group_width = heads_per_group * head_dim;
    let width = num_q_heads * head_dim;
    if group_width == 0 || q_rope.len() < width || attn_out.len() < width {
        return Err(ModelError::ShapeInvariant {
            tensor: format!("layer {layer_idx}: attention buffers"),
            expected: format!("a nonzero GQA group and >= {width} floats of queries and output"),
            actual: format!(
                "group {heads_per_group} x head_dim {head_dim}, q {} / out {}",
                q_rope.len(),
                attn_out.len()
            ),
        });
    }
    let (queries, outputs) = (&q_rope[..width], &mut attn_out[..width]);
    if parallel {
        outputs
            .par_chunks_mut(group_width)
            .zip(queries.par_chunks(group_width))
            .enumerate()
            .try_for_each(|(kv_head, (out_group, q_group))| {
                kv_cache.attend_group(layer_idx, kv_head, seq_len, q_group, out_group)
            })
    } else {
        for (kv_head, (out_group, q_group)) in outputs
            .chunks_mut(group_width)
            .zip(queries.chunks(group_width))
            .enumerate()
        {
            kv_cache.attend_group(layer_idx, kv_head, seq_len, q_group, out_group)?;
        }
        Ok(())
    }
}

/// Validate the shape invariants a bad config or a corrupted checkpoint
/// could otherwise violate silently (M-12).
///
/// `KvCache` is allocated once for the whole model with a single fixed
/// `(num_kv_heads, head_dim, max_seq_len)` stride shared by every layer's
/// `cache_offset(layer, head, pos)` (`((layer * num_kv_heads + head) *
/// max_seq_len + pos) * head_dim`). A layer whose own geometry disagrees
/// with that stride does not error there — it silently reads or writes a
/// different layer's or a different head's slot. This function is called at
/// the very top of every `forward*` entry point, before any KV-cache access
/// and before `heads_per_group = num_heads / num_kv_heads` (which panics on
/// a zero divisor), so a bad shape fails loudly instead of aliasing.
///
/// Checks, in an order chosen so no earlier check can itself panic:
/// 1. `num_kv_heads > 0` — a zero divisor would panic at the GQA grouping
///    computation immediately after this call returns.
/// 2. `num_heads % num_kv_heads == 0` — GQA grouping must be exact.
/// 3. `attn_q`'s output width is `head_dim * num_heads` (a plain Q
///    projection) or exactly double that (the Bonsai-2 q|gate interleave,
///    B2-11 — not yet implemented on this `TransformerBlock`, but the check
///    is written to already accept that layout).
/// 4. This layer's `(head_dim, num_kv_heads)` matches the cache's fixed
///    stride — the direct fix for the "silent cross-layer KV read" defect.
/// 5. `layer_idx` is within the cache's allocated layer count.
/// 6. `pos` is within the cache's allocated sequence length (the "`kv_slot`
///    in range" check).
#[allow(clippy::too_many_arguments)]
pub(super) fn validate_shapes(
    layer_idx: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q_out_features: usize,
    kv_cache: &KvCache,
    pos: usize,
) -> ModelResult<()> {
    if num_kv_heads == 0 {
        return Err(ModelError::ShapeInvariant {
            tensor: format!("layer {layer_idx}: num_kv_heads"),
            expected: "> 0".to_string(),
            actual: "0".to_string(),
        });
    }
    if !num_heads.is_multiple_of(num_kv_heads) {
        return Err(ModelError::ShapeInvariant {
            tensor: format!("layer {layer_idx}: num_attention_heads % num_kv_heads"),
            expected: "0".to_string(),
            actual: (num_heads % num_kv_heads).to_string(),
        });
    }
    let plain = head_dim * num_heads;
    if q_out_features != plain && q_out_features != plain * 2 {
        return Err(ModelError::ShapeInvariant {
            tensor: format!("layer {layer_idx}: attn_q.out_features"),
            expected: format!(
                "{plain} (head_dim*num_heads) or {} (2x, Bonsai-2 q|gate split)",
                plain * 2
            ),
            actual: q_out_features.to_string(),
        });
    }
    if head_dim != kv_cache.head_dim() || num_kv_heads != kv_cache.num_kv_heads() {
        return Err(ModelError::ShapeInvariant {
            tensor: format!("layer {layer_idx}: (head_dim, num_kv_heads) vs KvCache stride"),
            expected: format!("({}, {})", kv_cache.head_dim(), kv_cache.num_kv_heads()),
            actual: format!("({head_dim}, {num_kv_heads})"),
        });
    }
    if layer_idx >= kv_cache.num_layers() {
        return Err(ModelError::ShapeMismatch {
            name: "layer_idx vs KvCache num_layers".to_string(),
            expected: vec![kv_cache.num_layers()],
            actual: vec![layer_idx],
        });
    }
    if pos >= kv_cache.max_seq_len() {
        return Err(ModelError::SequenceTooLong {
            seq_len: pos + 1,
            max_ctx: kv_cache.max_seq_len(),
        });
    }
    Ok(())
}

/// Advance `kv_cache`'s internal sequence-length cursor to cover `pos`
/// (M-19).
///
/// The production forward path never called `KvCache::advance()`, so
/// `seq_len()`/`utilization_ratio()` were permanently stale even though the
/// hot path already derives its own correct LOCAL `seq_len = pos + 1` for
/// the attention span itself (that value never depended on the cursor).
/// `advance()` increments by exactly one and is not idempotent, but
/// `TransformerBlock::forward*` runs once PER LAYER per generated token, all
/// sharing the same `KvCache` — calling it unconditionally on every call
/// would overcount by a factor of `num_layers`. This loop is a monotonic
/// "catch up to `pos + 1`" that only ever advances forward: it does the
/// real work on whichever layer happens to run first for a given token and
/// is a safe no-op for every later layer's call in that same step, and it
/// can never clobber a rewind that speculative decoding performs via
/// `KvCache::set_seq_len`/`truncate` (those move the cursor down or to
/// exactly `pos`, never past it, so a later `forward*` call for the same or
/// a later `pos` only ever advances from there). Call only after
/// `validate_shapes` has confirmed `pos < max_seq_len()`, which bounds this
/// loop to at most `max_seq_len()` iterations.
pub(super) fn advance_kv_cache_to(kv_cache: &mut KvCache, pos: usize) {
    while kv_cache.seq_len() <= pos {
        kv_cache.advance();
    }
}

/// Attempt a fused ternary GEMV via direct Metal dispatch (M-21).
///
/// Concatenates the given ternary (`BlockTQ2_0_g128`) AoS byte blobs into
/// one SoA-uploaded buffer keyed by `weight_slot` (reformatted and cached on
/// first call, reused after) and runs a single GEMV producing all `n_rows`
/// outputs in one command buffer — the ternary mirror of
/// [`oxibonsai_kernels::try_metal_qkv`], which exists only for the
/// `Q1_0_g128` format (fused handles were built only for 1-bit weights in
/// `upload.rs`, so ternary models paid one GPU dispatch per projection).
/// Generalized over `aos_parts` (rather than fixed at three) so the same
/// helper could serve a future 2-way gate+up fusion too, though today only
/// the 3-way Q+K+V call site uses it.
///
/// `weight_slot` must be a stable, per-instance identity, unique for the
/// lifetime of the process. Call sites pass `attn_q.gpu_handle()`'s own
/// `.id()` — the per-matrix `GpuWeightHandle` that `LinearTernary`'s own
/// `upload_to_gpu()` already populates via `TernaryKernel::upload_weights_ternary`
/// (see `upload.rs`) — rather than a raw pointer into this model's mmap.
/// That id is drawn from the same process-global monotonic counter
/// (`NEXT_HANDLE_ID.fetch_add(1)` in `scirs2_backend.rs`) that 1-bit fused
/// handles use, so it is never reused across model loads. This matters
/// because `MetalGraph::get_or_upload_tq2_weight_soa_lazy` caches under
/// `LEGACY_MODEL_EPOCH` (`metal_full_layer/types.rs`, no per-model-epoch
/// component) and is never reclaimed by `MetalGraph::release_model()` for
/// that epoch: keying on the weight's own `as_ptr()` (an earlier version of
/// this function did) would let a model loaded after a previous one was
/// dropped, whose mmap happens to land on a freed address, be served the
/// PRIOR model's fused Q|K|V weights under the same cache slot — silently
/// wrong logits. A monotonically-issued id cannot collide from address
/// reuse the way `as_ptr()` could, but it is not collision-proof forever:
/// the other `WeightKind::Tq2Soa` slots in the process are `3_000_000`/
/// `6_000_000 + layer*10` (`model/types/forward_metal.rs`) and
/// `oxibonsai-image`'s own pointer-keyed cache, so this mitigation holds
/// exactly **while `NEXT_HANDLE_ID` stays below `3_000_000`** — a
/// long-lived server process that loads/unloads enough models to walk the
/// counter past that point would reopen the same silent-aliasing class
/// this fix removes, now against `forward_metal.rs`'s fixed slots instead
/// of a freed mmap address. Closing that gap for good (unbounded, not just
/// below the current fixed slots) is MET-02's `WeightKey { model_epoch, .. }`
/// work, which this is a same-epoch mitigation for, not a replacement of.
///
/// Returns `Err` (never panics) if Metal is unavailable or the dispatch
/// fails; callers fall back to the per-matrix path in that case.
///
/// # Known cost: double GPU residency for Q/K/V
///
/// `upload_to_gpu()` already gave each of `attn_q`/`attn_k`/`attn_v` its own
/// per-matrix GPU buffer in `Scirs2Backend`'s weight cache (that is exactly
/// the `GpuWeightHandle` this function keys on above); this function then
/// uploads a *second*, concatenated copy of the same three matrices into
/// `MetalGraph`'s independent cache the first time it runs for a given
/// `weight_slot`. Under `LEGACY_MODEL_EPOCH` that second copy is never
/// reclaimed for the life of the process, so a ternary model pays roughly
/// 2x GPU memory for every layer's Q/K/V — material at the 27B target this
/// helper exists for. `Scirs2Backend` has no per-handle eviction API (only
/// a clear-everything `clear_weight_cache()`, which would also drop
/// `attn_output`'s and any FFN matrices' own per-matrix ternary handles
/// still needed by the non-fused paths), so freeing just the Q/K/V copies
/// once the fused path is confirmed live would need a new eviction primitive
/// in `oxibonsai-kernels` — out of this file's reach. Documented here rather
/// than fixed; a real fix belongs alongside MET-02's epoch plumbing.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub(super) fn try_metal_gemv_ternary_fused(
    input: &[f32],
    output: &mut [f32],
    weight_slot: u64,
    aos_parts: &[&[u8]],
    n_rows: usize,
    k: usize,
) -> Result<(), oxibonsai_kernels::MetalGraphError> {
    let graph = oxibonsai_kernels::MetalGraph::global()?;
    let weight = graph.get_or_upload_tq2_weight_soa_lazy(weight_slot, || {
        let total: usize = aos_parts.iter().map(|part| part.len()).sum();
        let mut fused = Vec::with_capacity(total);
        for part in aos_parts {
            fused.extend_from_slice(part);
        }
        fused
    })?;
    graph.encode_gemv_tq2(&weight, input, output, n_rows, k)
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
    /// Build `n` ternary blocks with a fixed pattern byte.
    ///
    /// `pattern` must not encode the reserved 2-bit code `0b11` in any lane
    /// (that combination means "+2" in `PQ2_0` and is rejected on sight by
    /// `MetalGraph::upload_tq2_weight_soa`'s `validate_tq2_ternary_codes`
    /// screen when a test exercises the real Metal path). `0x00` (every lane
    /// `0b00` = -1) is always safe.
    fn make_ternary_blocks(
        n: usize,
        scale: f32,
        pattern: u8,
    ) -> Vec<oxibonsai_core::BlockTQ2_0_g128> {
        (0..n)
            .map(|_| oxibonsai_core::BlockTQ2_0_g128 {
                qs: [pattern; 32],
                d: f16::from_f32(scale),
            })
            .collect()
    }
    /// Create a minimal ternary-weighted test block with the given
    /// dimensions (M-21: mirrors `make_test_block`, but with `LinearTernary`
    /// weights so the fused-ternary-GPU-handle path is exercisable).
    #[allow(clippy::too_many_arguments)]
    fn make_ternary_test_block<'a>(
        h: usize,
        hd: usize,
        nq: usize,
        nkv: usize,
        inter: usize,
        kernel: std::sync::Arc<oxibonsai_kernels::KernelDispatcher>,
        q_blocks: &'a [oxibonsai_core::BlockTQ2_0_g128],
        k_blocks: &'a [oxibonsai_core::BlockTQ2_0_g128],
        v_blocks: &'a [oxibonsai_core::BlockTQ2_0_g128],
        o_blocks: &'a [oxibonsai_core::BlockTQ2_0_g128],
        gate_blocks: &'a [oxibonsai_core::BlockTQ2_0_g128],
        up_blocks: &'a [oxibonsai_core::BlockTQ2_0_g128],
        down_blocks: &'a [oxibonsai_core::BlockTQ2_0_g128],
    ) -> TransformerBlock<'a> {
        TransformerBlock::new(
            0,
            RmsNorm::new(vec![1.0; h], 1e-6),
            LinearTernary::new(q_blocks, nq * hd, h, kernel.clone())
                .expect("q")
                .into(),
            LinearTernary::new(k_blocks, nkv * hd, h, kernel.clone())
                .expect("k")
                .into(),
            LinearTernary::new(v_blocks, nkv * hd, h, kernel.clone())
                .expect("v")
                .into(),
            LinearTernary::new(o_blocks, h, nq * hd, kernel.clone())
                .expect("o")
                .into(),
            RmsNorm::new(vec![1.0; hd], 1e-6),
            RmsNorm::new(vec![1.0; hd], 1e-6),
            RmsNorm::new(vec![1.0; h], 1e-6),
            LinearTernary::new(gate_blocks, inter, h, kernel.clone())
                .expect("gate")
                .into(),
            LinearTernary::new(up_blocks, inter, h, kernel.clone())
                .expect("up")
                .into(),
            LinearTernary::new(down_blocks, h, inter, kernel)
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

    // ── M-12: shape-invariant validation ────────────────────────────────

    /// Reproduces the M-12 defect: `attention.head_count_kv = 0` is
    /// constructible (an empty `Linear1Bit` and an empty `KvCache` both
    /// accept it) and previously reached `forward.rs`'s
    /// `heads_per_group = nq / nkv` before any validation ran, panicking
    /// with a divide-by-zero instead of failing gracefully.
    #[test]
    fn forward_rejects_zero_kv_heads_instead_of_panicking() {
        let (h, hd, nq, nkv, inter) = (128, 64, 2, 0, 256);
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
        let result = block.forward(&mut hidden, 0, &mut kv_cache, &rope, kernel.as_ref());
        assert!(
            result.is_err(),
            "a zero-kv-head config must return an error, not silently divide-by-zero panic"
        );
    }

    #[test]
    fn forward_rejects_non_divisible_head_counts() {
        // 4 query heads cannot be evenly grouped over 3 KV heads.
        // (nq * hd == h, matching `make_test_block`'s implicit assumption
        // that `attn_output`'s in_features equals hidden_size; h and
        // nq * hd both stay multiples of 128 for `Linear1Bit`.)
        let (h, hd, nq, nkv, inter) = (256, 64, 4, 3, 256);
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
        let result = block.forward(&mut hidden, 0, &mut kv_cache, &rope, kernel.as_ref());
        match result {
            Err(ModelError::ShapeInvariant { .. }) => {}
            other => panic!("expected ShapeInvariant for 3 % 2 != 0, got {other:?}"),
        }
    }

    #[test]
    fn forward_rejects_q_projection_width_mismatch() {
        // attn_q is built with one head too many, disagreeing with the
        // `num_heads`/`head_dim` the block itself is told to use.
        let (h, hd, nq, nkv, inter) = (128, 64, 2, 1, 256);
        let bpr = h / 128;
        let wrong_nq = nq + 1;
        let q_b = make_blocks(wrong_nq * hd * bpr, 0.01, 0xFF);
        let k_b = make_blocks(nkv * hd * bpr, 0.01, 0xFF);
        let v_b = make_blocks(nkv * hd * bpr, 0.01, 0xFF);
        let o_b = make_blocks(h * bpr, 0.01, 0xFF);
        let g_b = make_blocks(inter * bpr, 0.01, 0xFF);
        let u_b = make_blocks(inter * bpr, 0.01, 0xFF);
        let d_b = make_blocks(h * (inter / 128), 0.01, 0xFF);
        let kernel = std::sync::Arc::new(oxibonsai_kernels::KernelDispatcher::auto_detect());
        let block = TransformerBlock::new(
            0,
            RmsNorm::new(vec![1.0; h], 1e-6),
            Linear1Bit::new(&q_b, wrong_nq * hd, h, kernel.clone())
                .expect("q")
                .into(),
            Linear1Bit::new(&k_b, nkv * hd, h, kernel.clone())
                .expect("k")
                .into(),
            Linear1Bit::new(&v_b, nkv * hd, h, kernel.clone())
                .expect("v")
                .into(),
            Linear1Bit::new(&o_b, h, nq * hd, kernel.clone())
                .expect("o")
                .into(),
            RmsNorm::new(vec![1.0; hd], 1e-6),
            RmsNorm::new(vec![1.0; hd], 1e-6),
            RmsNorm::new(vec![1.0; h], 1e-6),
            Linear1Bit::new(&g_b, inter, h, kernel.clone())
                .expect("gate")
                .into(),
            Linear1Bit::new(&u_b, inter, h, kernel.clone())
                .expect("up")
                .into(),
            Linear1Bit::new(&d_b, h, inter, kernel.clone())
                .expect("down")
                .into(),
            nq,
            nkv,
            hd,
            h,
        );
        let rope = RopeTable::new(hd, 16, 10000.0);
        let mut kv_cache = KvCache::new(1, nkv, hd, 16);
        let mut hidden: Vec<f32> = (0..h).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let result = block.forward(&mut hidden, 0, &mut kv_cache, &rope, kernel.as_ref());
        match result {
            Err(ModelError::ShapeInvariant { .. }) => {}
            other => panic!("expected ShapeInvariant for attn_q width mismatch, got {other:?}"),
        }
    }

    #[test]
    fn forward_rejects_kv_cache_head_dim_mismatch() {
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
        // Cache stride disagrees with the block's own head_dim: exactly the
        // "silent cross-layer KV read" precondition M-12 describes.
        let mut kv_cache = KvCache::new(1, nkv, hd + 1, 16);
        let mut hidden: Vec<f32> = (0..h).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let result = block.forward(&mut hidden, 0, &mut kv_cache, &rope, kernel.as_ref());
        match result {
            Err(ModelError::ShapeInvariant { .. }) => {}
            other => panic!("expected ShapeInvariant for KvCache head_dim mismatch, got {other:?}"),
        }
    }

    #[test]
    fn forward_rejects_layer_idx_beyond_cache_num_layers() {
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
        // `make_test_block` always builds layer_idx = 0.
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
        let mut kv_cache = KvCache::new(0, nkv, hd, 16); // zero layers allocated
        let mut hidden: Vec<f32> = (0..h).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let result = block.forward(&mut hidden, 0, &mut kv_cache, &rope, kernel.as_ref());
        match result {
            Err(ModelError::ShapeMismatch { .. }) => {}
            other => panic!("expected ShapeMismatch for layer_idx >= num_layers, got {other:?}"),
        }
    }

    #[test]
    fn forward_rejects_pos_beyond_max_seq_len() {
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
        let mut kv_cache = KvCache::new(1, nkv, hd, 4); // max_seq_len = 4
        let mut hidden: Vec<f32> = (0..h).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let result = block.forward(&mut hidden, 10, &mut kv_cache, &rope, kernel.as_ref());
        match result {
            Err(ModelError::SequenceTooLong { seq_len, max_ctx }) => {
                assert_eq!(seq_len, 11);
                assert_eq!(max_ctx, 4);
            }
            other => panic!("expected SequenceTooLong for pos >= max_seq_len, got {other:?}"),
        }
    }

    // ── M-19: KvCache::advance() wiring ─────────────────────────────────

    /// Simulates `BonsaiModel::forward`'s per-token loop over every layer,
    /// all sharing one `KvCache`: `seq_len()` must land on exactly
    /// `pos + 1` after ALL layers ran for that token, never
    /// `num_layers * (pos + 1)` (which a naive unconditional `advance()`
    /// call in every layer's `forward()` would produce).
    #[test]
    fn forward_advances_kv_cache_seq_len_once_per_token_across_layers() {
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
        let num_layers = 3usize;
        let mut kv_cache = KvCache::new(num_layers, nkv, hd, 16);
        let rope = RopeTable::new(hd, 16, 10000.0);
        let blocks: Vec<TransformerBlock> = (0..num_layers)
            .map(|layer_idx| {
                TransformerBlock::new(
                    layer_idx,
                    RmsNorm::new(vec![1.0; h], 1e-6),
                    Linear1Bit::new(&q_b, nq * hd, h, kernel.clone())
                        .expect("q")
                        .into(),
                    Linear1Bit::new(&k_b, nkv * hd, h, kernel.clone())
                        .expect("k")
                        .into(),
                    Linear1Bit::new(&v_b, nkv * hd, h, kernel.clone())
                        .expect("v")
                        .into(),
                    Linear1Bit::new(&o_b, h, nq * hd, kernel.clone())
                        .expect("o")
                        .into(),
                    RmsNorm::new(vec![1.0; hd], 1e-6),
                    RmsNorm::new(vec![1.0; hd], 1e-6),
                    RmsNorm::new(vec![1.0; h], 1e-6),
                    Linear1Bit::new(&g_b, inter, h, kernel.clone())
                        .expect("gate")
                        .into(),
                    Linear1Bit::new(&u_b, inter, h, kernel.clone())
                        .expect("up")
                        .into(),
                    Linear1Bit::new(&d_b, h, inter, kernel.clone())
                        .expect("down")
                        .into(),
                    nq,
                    nkv,
                    hd,
                    h,
                )
            })
            .collect();
        assert_eq!(kv_cache.seq_len(), 0);
        for pos in 0..4usize {
            let mut hidden: Vec<f32> = (0..h).map(|i| (i as f32 + 1.0) * 0.01).collect();
            for block in &blocks {
                block
                    .forward(&mut hidden, pos, &mut kv_cache, &rope, kernel.as_ref())
                    .expect("forward should succeed");
            }
            assert_eq!(
                kv_cache.seq_len(),
                pos + 1,
                "seq_len should be exactly pos+1 after all {num_layers} layers ran for \
                 position {pos}, not overcounted by the number of layers"
            );
        }
    }

    // ── M-17: sliding-window parity ─────────────────────────────────────

    /// A window at least as large as the sequence must attend to exactly
    /// the same positions, in the same order, as plain (unwindowed)
    /// attention — so the two entry points must produce bit-identical
    /// output. Neither block calls `upload_to_gpu`, so `fused_qkv_handle`
    /// stays `None` and both `forward()` and `forward_with_sliding_window()`
    /// take the identical CPU code path (no Metal/CUDA full-layer fast path
    /// to introduce an unrelated numerical difference).
    #[test]
    fn sliding_window_with_window_ge_seq_len_matches_full_attention() {
        // nq * hd == h, matching `make_test_block`'s implicit assumption
        // that `attn_output`'s in_features equals hidden_size.
        let (h, hd, nq, nkv, inter) = (256, 64, 4, 2, 256);
        let bpr = h / 128;
        let q_b = make_blocks(nq * hd * bpr, 0.01, 0xAB);
        let k_b = make_blocks(nkv * hd * bpr, 0.01, 0xCD);
        let v_b = make_blocks(nkv * hd * bpr, 0.01, 0xEF);
        let o_b = make_blocks(h * bpr, 0.01, 0x12);
        let g_b = make_blocks(inter * bpr, 0.01, 0x34);
        let u_b = make_blocks(inter * bpr, 0.01, 0x56);
        let d_b = make_blocks(h * (inter / 128), 0.01, 0x78);
        let kernel = std::sync::Arc::new(oxibonsai_kernels::KernelDispatcher::auto_detect());

        let full_block = make_test_block(
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
        let sw_block = make_test_block(
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
        // Window (32) plus sink (4) is far larger than the 5-token sequence
        // this test runs, so `attention_range` returns every position
        // 0..seq_len in order for every step below — the same set, in the
        // same order, that plain `compute_gqa_attention` uses.
        let sw_config = SlidingWindowConfig::new(32, 4);
        let rope = RopeTable::new(hd, 16, 10000.0);
        let mut full_kv_cache = KvCache::new(1, nkv, hd, 16);
        let mut sw_kv_cache = KvCache::new(1, nkv, hd, 16);

        for pos in 0..5usize {
            let mut full_hidden: Vec<f32> =
                (0..h).map(|i| ((i + pos) as f32 + 1.0) * 0.01).collect();
            let mut sw_hidden = full_hidden.clone();
            full_block
                .forward(
                    &mut full_hidden,
                    pos,
                    &mut full_kv_cache,
                    &rope,
                    kernel.as_ref(),
                )
                .expect("full attention forward should succeed");
            sw_block
                .forward_with_sliding_window(
                    &mut sw_hidden,
                    pos,
                    &mut sw_kv_cache,
                    &rope,
                    kernel.as_ref(),
                    Some(&sw_config),
                )
                .expect("sliding-window forward should succeed");
            for (i, (a, b)) in full_hidden.iter().zip(sw_hidden.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-5,
                    "position {pos} element {i}: full={a} sw={b} \
                     (window >= seq_len must match full attention exactly)"
                );
            }
        }
    }

    // ── M-21: fused ternary GPU handles ──────────────────────────────────

    /// `forward()` must produce the same result whether or not
    /// `upload_to_gpu()` ran first: without it, every projection uses the
    /// CPU/adaptive ternary SIMD path; with it, `attn_q`/`attn_k`/`attn_v`
    /// (and the FFN matrices) are individually GPU-cached AND, on a Metal
    /// build, the fused ternary QKV path in `forward.rs` may additionally
    /// engage (`try_metal_gemv_ternary_fused`). Either way the numerical
    /// result must agree with the CPU reference to a tight tolerance.
    #[test]
    fn ternary_forward_matches_with_and_without_gpu_upload() {
        let (h, hd, nq, nkv, inter) = (256, 64, 4, 2, 512);
        let bpr = h / 128;
        // Pattern 0x00 -> every 2-bit lane is 0b00 (-1): never the reserved
        // 0b11 ("+2", a PQ2_0-only code) that the Metal TQ2 uploader
        // rejects on sight for a genuinely-ternary tensor.
        let q_b = make_ternary_blocks(nq * hd * bpr, 0.03, 0x00);
        let k_b = make_ternary_blocks(nkv * hd * bpr, 0.03, 0x00);
        let v_b = make_ternary_blocks(nkv * hd * bpr, 0.03, 0x00);
        let o_b = make_ternary_blocks(h * bpr, 0.03, 0x00);
        let g_b = make_ternary_blocks(inter * bpr, 0.03, 0x00);
        let u_b = make_ternary_blocks(inter * bpr, 0.03, 0x00);
        let d_b = make_ternary_blocks(h * (inter / 128), 0.03, 0x00);
        let kernel = std::sync::Arc::new(oxibonsai_kernels::KernelDispatcher::auto_detect());

        let reference_block = make_ternary_test_block(
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
        let mut ref_kv_cache = KvCache::new(1, nkv, hd, 16);
        let mut ref_hidden: Vec<f32> = (0..h).map(|i| (i as f32 + 1.0) * 0.001).collect();
        let original = ref_hidden.clone();
        reference_block
            .forward(
                &mut ref_hidden,
                0,
                &mut ref_kv_cache,
                &rope,
                kernel.as_ref(),
            )
            .expect("reference (no GPU upload) forward should succeed");
        let ref_max_diff = ref_hidden
            .iter()
            .zip(original.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            ref_max_diff > 1e-8,
            "reference forward should modify hidden state, max_diff={ref_max_diff}"
        );

        let mut gpu_block = make_ternary_test_block(
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
        gpu_block.upload_to_gpu(kernel.as_ref());
        let mut gpu_kv_cache = KvCache::new(1, nkv, hd, 16);
        let mut gpu_hidden: Vec<f32> = (0..h).map(|i| (i as f32 + 1.0) * 0.001).collect();
        gpu_block
            .forward(
                &mut gpu_hidden,
                0,
                &mut gpu_kv_cache,
                &rope,
                kernel.as_ref(),
            )
            .expect("GPU-uploaded forward should succeed");

        let max_diff = ref_hidden
            .iter()
            .zip(gpu_hidden.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-3,
            "ternary forward must agree whether or not upload_to_gpu() ran \
             (fused Metal fast path vs per-matrix path), max_diff={max_diff}"
        );
    }

    /// Blocking-fix verification for the `weight_slot` rekey.
    ///
    /// `ternary_forward_matches_with_and_without_gpu_upload` above cannot by
    /// itself prove the fused Metal dispatch actually ran: every caller of
    /// `try_metal_gemv_ternary_fused` silently falls back to the per-matrix
    /// path on `Err`, so a test that only checks "same answer either way"
    /// would still pass if the helper (or `attn_q.gpu_handle()`) were
    /// silently dead on the test machine. This test calls
    /// `try_metal_gemv_ternary_fused` directly with a real
    /// `GpuWeightHandle::id()` — the exact value the fixed call sites in
    /// `forward.rs`/`forward_stats.rs`/`forward_sw.rs` now pass as
    /// `weight_slot` — and `.expect()`s success, so a dead or broken Metal
    /// path fails loudly here instead of being masked by a fallback.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn try_metal_gemv_ternary_fused_engages_and_matches_cpu_reference() {
        use oxibonsai_kernels::TernaryKernel;

        let (h, hd, nq, nkv) = (256, 64, 4, 2);
        let bpr = h / 128;
        let q_b = make_ternary_blocks(nq * hd * bpr, 0.05, 0x00);
        let k_b = make_ternary_blocks(nkv * hd * bpr, 0.05, 0x00);
        let v_b = make_ternary_blocks(nkv * hd * bpr, 0.05, 0x00);

        let kernel = oxibonsai_kernels::KernelDispatcher::auto_detect();
        // The same call `LinearTernary::upload_to_gpu()` makes in
        // production; its `.id()` is what the fixed call sites now key
        // `try_metal_gemv_ternary_fused`'s `weight_slot` on.
        let handle = kernel.upload_weights_ternary(&q_b).expect(
            "this test requires a real GPU tier (metal feature + Metal-capable \
             hardware) to actually verify the fused ternary dispatch; if this \
             fails, `try_metal_gemv_ternary_fused`'s call sites are dead on \
             this machine regardless of the blocking fix",
        );

        let input: Vec<f32> = (0..h).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let q_rows = nq * hd;
        let k_rows = nkv * hd;
        let total_rows = q_rows + k_rows + k_rows;
        let mut fused_output = vec![0.0f32; total_rows];
        let q_bytes = blocks_as_bytes_ternary(&q_b);
        let k_bytes = blocks_as_bytes_ternary(&k_b);
        let v_bytes = blocks_as_bytes_ternary(&v_b);

        try_metal_gemv_ternary_fused(
            &input,
            &mut fused_output,
            handle.id(),
            &[q_bytes, k_bytes, v_bytes],
            total_rows,
            h,
        )
        .expect(
            "fused ternary Metal GEMV must actually succeed on Metal-capable \
             hardware, not silently no-op",
        );

        // Cross-check against the plain CPU reference GEMV: the fused
        // kernel must produce the identical `[Q | K | V]` concatenation.
        let mut q_ref = vec![0.0f32; q_rows];
        let mut k_ref = vec![0.0f32; k_rows];
        let mut v_ref = vec![0.0f32; k_rows];
        kernel
            .gemv_ternary_g128(&q_b, &input, &mut q_ref, q_rows, h)
            .expect("CPU reference Q GEMV");
        kernel
            .gemv_ternary_g128(&k_b, &input, &mut k_ref, k_rows, h)
            .expect("CPU reference K GEMV");
        kernel
            .gemv_ternary_g128(&v_b, &input, &mut v_ref, k_rows, h)
            .expect("CPU reference V GEMV");

        for (i, (a, b)) in fused_output[..q_rows].iter().zip(q_ref.iter()).enumerate() {
            assert!((a - b).abs() < 1e-3, "Q[{i}]: fused={a} cpu_ref={b}");
        }
        for (i, (a, b)) in fused_output[q_rows..q_rows + k_rows]
            .iter()
            .zip(k_ref.iter())
            .enumerate()
        {
            assert!((a - b).abs() < 1e-3, "K[{i}]: fused={a} cpu_ref={b}");
        }
        for (i, (a, b)) in fused_output[q_rows + k_rows..total_rows]
            .iter()
            .zip(v_ref.iter())
            .enumerate()
        {
            assert!((a - b).abs() < 1e-3, "V[{i}]: fused={a} cpu_ref={b}");
        }
    }
}
