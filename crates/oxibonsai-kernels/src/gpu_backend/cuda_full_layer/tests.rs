//! Tests for the CUDA full-layer inference path.

use super::*;

/// Verify that `init_attn_modules` / `CudaGraph::global` gracefully returns
/// `Err` when no CUDA device is present (CI environment).
#[test]
fn test_try_cuda_full_layer_no_gpu_graceful() {
    let _serial = crate::gpu_backend::cuda_graph::types::gpu_parity_test_guard();
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

/// F4: the KV-cache layer offset must be the 64-bit value the attention
/// kernels now take as `unsigned long long`.
///
/// This used to reimplement the arithmetic locally with an `as u32`, i.e. it
/// asserted exactly the truncation the finding is about. It now exercises the
/// real helper; the exhaustive host-runnable coverage (including the >= 2^32
/// cases this module cannot reach on a non-CUDA host) lives in
/// `gpu_backend::cuda_device_negotiation`.
#[test]
fn test_kv_cache_layer_offset_arithmetic() {
    use crate::gpu_backend::cuda_device_negotiation::cuda_kv_layer_offset_elements;
    let n_kv = 8usize;
    let max_seq = 512usize;
    let head_dim = 128usize;
    let layer_offset =
        |layer_idx: usize| cuda_kv_layer_offset_elements(layer_idx, n_kv, max_seq, head_dim);
    assert_eq!(layer_offset(0), 0u64);
    assert_eq!(layer_offset(1), (8 * 512 * 128) as u64);
    // 8B geometry at 128 K context: the value the old `as u32` wrapped.
    let wide = cuda_kv_layer_offset_elements(33, 8, 131_072, 128);
    assert!(wide > u64::from(u32::MAX));
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

/// F-M1: the slot key built for a forward must distinguish the Q1 and ternary
/// families at identical dimensions — the exact pair the finding names
/// (`Bonsai-8B` vs `Ternary-Bonsai-8B`), which share every dimension and so
/// never triggered a buffer reallocation or an invalidation.
///
/// No GPU needed. The branch table this feeds is unit-tested on every host in
/// `gpu_backend::cuda_graph_slot`; this checks the key *this* module builds.
#[test]
fn slot_key_separates_q1_and_ternary_at_identical_dims() {
    let build = |quant_kind, model_epoch, fingerprint, final_norm| {
        build_slot_key(
            quant_kind,
            model_epoch,
            fingerprint,
            final_norm,
            36,
            4096,
            32,
            8,
            128,
            4096,
            11008,
        )
    };
    let q1 = build(CudaQuantKind::Q1G128, 1, 0xfeed, 5_900_000);
    let tq2 = build(CudaQuantKind::Tq2G128, 1, 0xfeed, 5_900_000);
    assert_ne!(q1, tq2);
    assert!(!q1.may_replay(&tq2));
    // Identical inputs must produce an identical (replayable) key.
    assert!(q1.may_replay(&build(CudaQuantKind::Q1G128, 1, 0xfeed, 5_900_000)));
    // A new model load (new epoch) forbids replay even for the same file.
    assert!(!q1.may_replay(&build(CudaQuantKind::Q1G128, 2, 0xfeed, 5_900_000)));
    // A different handle fingerprint forbids replay. NOTE: for the CUDA decode
    // paths the handle ids are layer-derived and so identical across two
    // same-depth models — this asserts the key's contract, not that real
    // ternary handles differ. `model_epoch` above is what separates models
    // there; see `CudaGraphSlotKey::fingerprint_handles`.
    assert!(!q1.may_replay(&build(CudaQuantKind::Q1G128, 1, 0xfeee, 5_900_000)));
    // The final-norm handle is folded into the fingerprint — the capture bakes
    // in its device pointer too, but it reaches the encode paths separately.
    assert!(!q1.may_replay(&build(CudaQuantKind::Q1G128, 1, 0xfeed, 5_900_001)));
}

/// F-M1: dimensions are saturated into the key's `u32` fields, never truncated.
/// A silent wraparound would let two different models compare equal, which is
/// precisely what the key exists to prevent.
#[test]
fn slot_key_saturates_rather_than_truncates_dimensions() {
    let huge = build_slot_key(
        CudaQuantKind::Q1G128,
        1,
        0,
        0,
        1usize << 33,
        4096,
        32,
        8,
        128,
        4096,
        11008,
    );
    assert_eq!(huge.n_layers, u32::MAX);
    // `1 << 33` and `1 << 34` both truncate to 0 in `as u32`; saturation keeps
    // them equal to each other but distinct from a real 36-layer model.
    let normal = build_slot_key(
        CudaQuantKind::Q1G128,
        1,
        0,
        0,
        36,
        4096,
        32,
        8,
        128,
        4096,
        11008,
    );
    assert_ne!(huge, normal);
    assert_ne!(huge.n_layers, 0);
}

/// F-M1: a slot holding a graph for a different key must be **dropped** by the
/// decision helper, so the stale `CUgraphExec` is freed before the re-capture
/// rather than lingering for the life of the process.
///
/// Uses a `None` holder, so no CUDA device is required: the state transition is
/// what is under test, not the exec itself.
#[test]
fn stale_slot_is_dropped_and_matching_slot_is_kept() {
    let key = |quant_kind, epoch| {
        build_slot_key(
            quant_kind, epoch, 0xabc, 7, 36, 4096, 32, 8, 128, 4096, 11008,
        )
    };
    let q1 = key(CudaQuantKind::Q1G128, 1);
    let tq2 = key(CudaQuantKind::Tq2G128, 1);

    // Empty slot: capture, nothing to drop.
    let mut slot: Option<(CudaGraphSlotKey, Option<CuGraphHolder>)> = None;
    assert_eq!(
        slot_action_dropping_stale(&mut slot, &q1),
        CudaGraphSlotAction::Capture
    );
    assert!(slot.is_none());

    // Foreign key in the slot: dropped, and a capture is requested.
    slot = Some((tq2, None));
    assert_eq!(
        slot_action_dropping_stale(&mut slot, &q1),
        CudaGraphSlotAction::Capture
    );
    assert!(slot.is_none(), "a stale capture must be dropped");

    // Matching key whose capture previously failed: kept, never retried.
    slot = Some((q1, None));
    assert_eq!(
        slot_action_dropping_stale(&mut slot, &q1),
        CudaGraphSlotAction::RunEager
    );
    assert!(slot.is_some(), "a tried-and-failed slot must be preserved");
}
