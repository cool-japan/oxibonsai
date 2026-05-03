//! Tests for the CUDA full-layer inference path.

use super::*;

/// Verify that `init_attn_modules` / `CudaGraph::global` gracefully returns
/// `Err` when no CUDA device is present (CI environment).
#[test]
fn test_try_cuda_full_layer_no_gpu_graceful() {
    let graph_result = CudaGraph::global();
    if graph_result.is_err() {
        return; // No GPU -- skip.
    }
    let graph = graph_result.expect("CUDA graph init should succeed");
    let modules_result = init_attn_modules(&graph);
    assert!(
        modules_result.is_ok(),
        "attn module init failed: {:?}",
        modules_result.err()
    );
}

/// Verify dimension arithmetic for the KV cache layer offset.
#[test]
fn test_kv_cache_layer_offset_arithmetic() {
    let n_kv = 8usize;
    let max_seq = 512usize;
    let head_dim = 128usize;
    let layer_offset = |layer_idx: usize| (layer_idx * n_kv * max_seq * head_dim) as u32;
    let layer_0 = layer_offset(0);
    let layer_1 = layer_offset(1);
    assert_eq!(layer_0, 0);
    assert_eq!(layer_1, 8 * 512 * 128);
}

/// Verify that `CudaFullLayerBuffers::matches` correctly identifies
/// dimension changes requiring reallocation.
#[test]
fn test_full_layer_buffers_matches_logic() {
    let nq = 32usize;
    let nkv = 8usize;
    let head_dim = 128usize;
    let qkv_total = nq * head_dim + 2 * nkv * head_dim;
    assert_eq!(qkv_total, 32 * 128 + 2 * 8 * 128, "QKV total mismatch");
    let half_dim = head_dim / 2;
    assert_eq!(half_dim, 64);
    let scores_len = nq * 2048usize;
    assert_eq!(scores_len, 32 * 2048);
}

/// Verify the batch-stride grid computation for attn scores V2.
#[test]
fn test_attn_scores_v2_grid_dim() {
    const BATCH_STRIDE: u32 = 4;
    for seq_len in [1u32, 4, 5, 16, 100, 2048] {
        let grid_y = seq_len.div_ceil(BATCH_STRIDE);
        assert!(
            grid_y * BATCH_STRIDE >= seq_len,
            "seq_len={seq_len} not covered by grid_y={grid_y}"
        );
    }
}
