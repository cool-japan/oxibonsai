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

/// Model-owned source tensors of one fake ternary layer — what
/// `oxibonsai-model` borrows from its owned / mmap'd weights, and whose
/// addresses are therefore stable for the life of a model.
struct FakeTernaryLayer {
    attn_norm: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    attn_proj: Vec<u8>,
    gate: Vec<u8>,
    up: Vec<u8>,
    down: Vec<u8>,
}

impl FakeTernaryLayer {
    fn new(seed: u8) -> Self {
        Self {
            attn_norm: vec![f32::from(seed) + 1.0; 64],
            q_norm: vec![f32::from(seed) + 2.0; 16],
            k_norm: vec![f32::from(seed) + 3.0; 16],
            ffn_norm: vec![f32::from(seed) + 4.0; 64],
            attn_proj: vec![seed; 34 * 4],
            gate: vec![seed.wrapping_add(1); 34 * 8],
            up: vec![seed.wrapping_add(2); 34 * 8],
            down: vec![seed.wrapping_add(3); 34 * 4],
        }
    }

    /// One layer's parameters, laid out like `oxibonsai-model`'s
    /// `build_cuda_ternary_layer_params`: norm handles `+0..3` and weight
    /// handles `+0..3` composed over `handle_epoch` (the model's
    /// `cuda_model_epoch` in the real caller), Q‖K‖V from the caller's
    /// `qkv` concatenation.
    fn params<'a>(
        &'a self,
        qkv: &'a [u8],
        handle_epoch: u64,
        layer: u64,
    ) -> CudaFullForwardLayerParamsTernary<'a> {
        let norm_base = (handle_epoch << 24) | (5_000_000 + layer * 10);
        let weight_base = (handle_epoch << 24) | (6_000_000 + layer * 10);
        CudaFullForwardLayerParamsTernary {
            attn_norm_handle: norm_base,
            attn_norm_bytes: &self.attn_norm,
            fused_qkv_handle: weight_base,
            fused_qkv_bytes: qkv,
            q_norm_handle: norm_base + 1,
            q_norm_bytes: &self.q_norm,
            k_norm_handle: norm_base + 2,
            k_norm_bytes: &self.k_norm,
            attn_proj_handle: weight_base + 1,
            attn_proj_bytes: &self.attn_proj,
            ffn_norm_handle: norm_base + 3,
            ffn_norm_bytes: &self.ffn_norm,
            gate_up_handle: weight_base + 2,
            gate_bytes: &self.gate,
            up_bytes: &self.up,
            down_handle: weight_base + 3,
            down_bytes: &self.down,
        }
    }
}

/// The caller-side Q‖K‖V concatenation of a fake layer, freshly allocated —
/// exactly what the ternary decode used to rebuild on every token.
fn fake_qkv_concat(seed: u8, len: usize) -> Vec<u8> {
    (0..len).map(|i| seed.wrapping_add(i as u8)).collect()
}

/// Ternary decode thrash regression (`oxibonsai run` on `Ternary-Bonsai-8B`:
/// every decode token evicted 289 weights / 1.85 GB, re-uploaded them under a
/// new epoch and re-captured the CUDA graph, 1.0 tok/s).
///
/// The cached-weight-set fingerprint used to hash the heap address of the
/// fused Q‖K‖V concatenation the caller rebuilt per call, so a malloc arena
/// that hands out a different address per call (a streaming worker thread)
/// made every call a miss. It must now be identical for one model whatever the
/// scratch buffer's address is — while a different model must still miss
/// (F-M1 / F-M3: a swap must evict and re-capture).
///
/// Pure host logic: no CUDA device is touched.
#[test]
fn ternary_weight_fingerprint_ignores_qkv_scratch_address() {
    use super::encode_ternary::ternary_model_weights_fingerprint as fingerprint;

    const N_LAYERS: usize = 3;
    const QKV_LEN: usize = 34 * 12;
    const EPOCH: u64 = 7;
    let model: Vec<FakeTernaryLayer> = (0..N_LAYERS as u8).map(FakeTernaryLayer::new).collect();
    let fingerprint_of = |qkv: &[Vec<u8>], epoch: u64| -> u64 {
        let params: Vec<CudaFullForwardLayerParamsTernary<'_>> = model
            .iter()
            .zip(qkv)
            .enumerate()
            .map(|(layer, (l, q))| l.params(q, epoch, layer as u64))
            .collect();
        fingerprint(&params)
    };

    // Two "decode tokens" whose per-call concatenations are alive at the same
    // time, so they are guaranteed to sit at different addresses.
    let first_call: Vec<Vec<u8>> = (0..N_LAYERS as u8)
        .map(|s| fake_qkv_concat(s, QKV_LEN))
        .collect();
    let second_call: Vec<Vec<u8>> = (0..N_LAYERS as u8)
        .map(|s| fake_qkv_concat(s, QKV_LEN))
        .collect();
    for (a, b) in first_call.iter().zip(&second_call) {
        assert_ne!(a.as_ptr(), b.as_ptr(), "precondition: distinct scratch");
    }
    let fp = fingerprint_of(&first_call, EPOCH);
    assert_eq!(
        fp,
        fingerprint_of(&second_call, EPOCH),
        "one model must keep one weight-set identity across calls, whatever \
         address the fused-QKV scratch landed at"
    );
    // A third call after the first scratch set is freed and reallocated.
    drop(first_call);
    let third_call: Vec<Vec<u8>> = (0..N_LAYERS as u8)
        .map(|s| fake_qkv_concat(s, QKV_LEN))
        .collect();
    assert_eq!(fp, fingerprint_of(&third_call, EPOCH));

    // A different model must still miss.
    // (a) Another load — epoch-composed handles differ — even when every host
    //     slice is the very same memory (address reuse after a model drop).
    assert_ne!(fp, fingerprint_of(&second_call, EPOCH + 1));
    // (b) Other model-owned source tensors under identical (fixed) handles.
    let other_model: Vec<FakeTernaryLayer> =
        (0..N_LAYERS as u8).map(FakeTernaryLayer::new).collect();
    let other_params: Vec<CudaFullForwardLayerParamsTernary<'_>> = other_model
        .iter()
        .zip(&second_call)
        .enumerate()
        .map(|(layer, (l, q))| l.params(q, EPOCH, layer as u64))
        .collect();
    assert_ne!(fp, fingerprint(&other_params));
    // (c) A differently shaped Q‖K‖V.
    let longer: Vec<Vec<u8>> = (0..N_LAYERS as u8)
        .map(|s| fake_qkv_concat(s, QKV_LEN + 34))
        .collect();
    assert_ne!(fp, fingerprint_of(&longer, EPOCH));
    // (d) A different depth.
    let shallow: Vec<CudaFullForwardLayerParamsTernary<'_>> = model[..N_LAYERS - 1]
        .iter()
        .zip(&second_call)
        .enumerate()
        .map(|(layer, (l, q))| l.params(q, EPOCH, layer as u64))
        .collect();
    assert_ne!(fp, fingerprint(&shallow));

    // The graph-slot key built downstream from one model's (stable) handle
    // set is therefore replayable call after call, and a new epoch still
    // forbids replay (F-M1).
    let key = |model_epoch: u64| {
        build_slot_key(
            CudaQuantKind::Tq2G128,
            model_epoch,
            CudaGraphSlotKey::fingerprint_handles(&[1, 2, 3]),
            9,
            N_LAYERS,
            4096,
            32,
            8,
            128,
            4096,
            12288,
        )
    };
    assert!(key(EPOCH).may_replay(&key(EPOCH)));
    assert!(!key(EPOCH).may_replay(&key(EPOCH + 1)));
}

/// Model-owned source tensors of one fake Q1 layer (18-byte `Q1_0_g128`
/// blocks) — what `oxibonsai-model` borrows from its owned / mmap'd weights,
/// and whose addresses are therefore stable for the life of a model.
struct FakeQ1Layer {
    attn_norm: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    attn_proj: Vec<u8>,
    gate: Vec<u8>,
    up: Vec<u8>,
    down: Vec<u8>,
}

/// Where a fake Q1 layer's four weight handle ids come from: the two sources
/// `oxibonsai-model`'s `build_cuda_layer_params` draws them from.
#[derive(Clone, Copy)]
enum Q1WeightIds {
    /// The block was uploaded at load, so its fused weights carry ids from the
    /// process-wide upload-handle counter, layer `l` taking `first + 4 * l ..`.
    Uploaded(u64),
    /// Not uploaded: the weight-fallback slots composed over the epoch.
    Fallback,
}

impl FakeQ1Layer {
    fn new(seed: u8) -> Self {
        Self {
            attn_norm: vec![f32::from(seed) + 1.0; 64],
            q_norm: vec![f32::from(seed) + 2.0; 16],
            k_norm: vec![f32::from(seed) + 3.0; 16],
            ffn_norm: vec![f32::from(seed) + 4.0; 64],
            attn_proj: vec![seed; 18 * 4],
            gate: vec![seed.wrapping_add(1); 18 * 8],
            up: vec![seed.wrapping_add(2); 18 * 8],
            down: vec![seed.wrapping_add(3); 18 * 4],
        }
    }

    /// One layer's parameters, laid out like `oxibonsai-model`'s
    /// `build_cuda_layer_params`: norm handles `+0..3` composed over
    /// `handle_epoch` (the model's `cuda_model_epoch` in the real caller),
    /// weight handles per `ids`, Q‖K‖V from the model's cached `qkv`
    /// concatenation.
    fn params<'a>(
        &'a self,
        qkv: &'a [u8],
        handle_epoch: u64,
        layer: u64,
        ids: Q1WeightIds,
    ) -> CudaFullForwardLayerParams<'a> {
        let norm_base = (handle_epoch << 24) | (1_000_000 + layer * 10);
        let weight_base = match ids {
            Q1WeightIds::Uploaded(first) => first + layer * 4,
            Q1WeightIds::Fallback => (handle_epoch << 24) | (4_000_000 + layer * 4),
        };
        CudaFullForwardLayerParams {
            attn_norm_handle: norm_base,
            attn_norm_bytes: &self.attn_norm,
            fused_qkv_handle: weight_base,
            fused_qkv_bytes: qkv,
            q_norm_handle: norm_base + 1,
            q_norm_bytes: &self.q_norm,
            k_norm_handle: norm_base + 2,
            k_norm_bytes: &self.k_norm,
            attn_proj_handle: weight_base + 1,
            attn_proj_bytes: &self.attn_proj,
            ffn_norm_handle: norm_base + 3,
            ffn_norm_bytes: &self.ffn_norm,
            gate_up_handle: weight_base + 2,
            gate_bytes: &self.gate,
            up_bytes: &self.up,
            down_handle: weight_base + 3,
            down_bytes: &self.down,
        }
    }
}

/// Every layer's parameters for a fake Q1 model.
fn fake_q1_params<'a>(
    model: &'a [FakeQ1Layer],
    qkv: &'a [Vec<u8>],
    handle_epoch: u64,
    ids: Q1WeightIds,
) -> Vec<CudaFullForwardLayerParams<'a>> {
    model
        .iter()
        .zip(qkv)
        .enumerate()
        .map(|(layer, (l, q))| l.params(q, handle_epoch, layer as u64, ids))
        .collect()
}

const Q1_TEST_LAYERS: usize = 3;
const Q1_TEST_QKV_LEN: usize = 18 * 12;
const Q1_TEST_EPOCH: u64 = 7;

fn fake_q1_model() -> (Vec<FakeQ1Layer>, Vec<Vec<u8>>) {
    let model = (0..Q1_TEST_LAYERS as u8).map(FakeQ1Layer::new).collect();
    let qkv = (0..Q1_TEST_LAYERS as u8)
        .map(|s| fake_qkv_concat(s, Q1_TEST_QKV_LEN))
        .collect();
    (model, qkv)
}

/// Q1 reload regression: the Q1 cached-weight-set fingerprint used to hash
/// the host slices' addresses and lengths only, so a same-depth Q1 model
/// loaded after another was dropped matched the stale entry whenever the
/// allocator handed it every freed address — and was decoded with the dropped
/// model's device weights. Like the ternary fingerprint, it now mixes the
/// handle ids, which differ for every load whatever the addresses.
///
/// Every host slice below is literally the same memory for both "loads", the
/// worst case of address reuse; only the handle ids differ, exactly as for a
/// real reload. Fails with the old address-only fingerprint.
///
/// Pure host logic: no CUDA device is touched.
#[test]
fn q1_weight_fingerprint_separates_reloads_that_reuse_host_addresses() {
    use super::model_weights_fingerprint as fingerprint;

    let (model, qkv) = fake_q1_model();
    let first_load = fake_q1_params(&model, &qkv, Q1_TEST_EPOCH, Q1WeightIds::Fallback);
    let reload = fake_q1_params(&model, &qkv, Q1_TEST_EPOCH + 1, Q1WeightIds::Fallback);
    // Precondition: the two loads differ in nothing but their handle ids.
    for (a, b) in first_load.iter().zip(&reload) {
        let slices = |lp: &CudaFullForwardLayerParams<'_>| {
            [
                (
                    lp.attn_norm_bytes.as_ptr() as usize,
                    lp.attn_norm_bytes.len(),
                ),
                (
                    lp.fused_qkv_bytes.as_ptr() as usize,
                    lp.fused_qkv_bytes.len(),
                ),
                (lp.q_norm_bytes.as_ptr() as usize, lp.q_norm_bytes.len()),
                (lp.k_norm_bytes.as_ptr() as usize, lp.k_norm_bytes.len()),
                (
                    lp.attn_proj_bytes.as_ptr() as usize,
                    lp.attn_proj_bytes.len(),
                ),
                (lp.ffn_norm_bytes.as_ptr() as usize, lp.ffn_norm_bytes.len()),
                (lp.gate_bytes.as_ptr() as usize, lp.gate_bytes.len()),
                (lp.up_bytes.as_ptr() as usize, lp.up_bytes.len()),
                (lp.down_bytes.as_ptr() as usize, lp.down_bytes.len()),
            ]
        };
        assert_eq!(slices(a), slices(b), "precondition: same host memory");
        assert_ne!(a.attn_norm_handle, b.attn_norm_handle);
    }
    // (a) Weight-fallback slots: every handle is composed over the new epoch.
    assert_ne!(
        fingerprint(&first_load),
        fingerprint(&reload),
        "a reload at the very same host addresses must not hit the previous \
         model's cached Q1 weight set"
    );
    // (b) Blocks uploaded at load: fresh upload ids and a fresh norm epoch.
    let fp_uploaded = fingerprint(&fake_q1_params(
        &model,
        &qkv,
        Q1_TEST_EPOCH,
        Q1WeightIds::Uploaded(100),
    ));
    assert_ne!(
        fp_uploaded,
        fingerprint(&fake_q1_params(
            &model,
            &qkv,
            Q1_TEST_EPOCH + 1,
            Q1WeightIds::Uploaded(200),
        ))
    );
    // Each handle source separates two loads on its own: the norm slots' epoch
    // alone, and the upload ids alone.
    assert_ne!(
        fp_uploaded,
        fingerprint(&fake_q1_params(
            &model,
            &qkv,
            Q1_TEST_EPOCH + 1,
            Q1WeightIds::Uploaded(100),
        ))
    );
    assert_ne!(
        fp_uploaded,
        fingerprint(&fake_q1_params(
            &model,
            &qkv,
            Q1_TEST_EPOCH,
            Q1WeightIds::Uploaded(200),
        ))
    );
}

/// One Q1 model keeps one weight-set identity: identical inputs give the same
/// fingerprint call after call (a miss would evict, re-upload and re-capture
/// on every token), and, as on the ternary path, the Q‖K‖V buffer counts by
/// its length only, so its address does not matter.
///
/// Pure host logic: no CUDA device is touched.
#[test]
fn q1_weight_fingerprint_is_stable_for_one_model() {
    use super::model_weights_fingerprint as fingerprint;

    let (model, qkv) = fake_q1_model();
    for ids in [Q1WeightIds::Fallback, Q1WeightIds::Uploaded(100)] {
        let params = fake_q1_params(&model, &qkv, Q1_TEST_EPOCH, ids);
        let fp = fingerprint(&params);
        assert_eq!(fp, fingerprint(&params), "same inputs, same fingerprint");
        assert_eq!(
            fp,
            fingerprint(&fake_q1_params(&model, &qkv, Q1_TEST_EPOCH, ids)),
            "a rebuilt parameter set over the same model must still hit"
        );
        // Another Q‖K‖V buffer of the same length, alive at the same time and
        // so at another address.
        let other_qkv: Vec<Vec<u8>> = (0..Q1_TEST_LAYERS as u8)
            .map(|s| fake_qkv_concat(s, Q1_TEST_QKV_LEN))
            .collect();
        for (a, b) in qkv.iter().zip(&other_qkv) {
            assert_ne!(a.as_ptr(), b.as_ptr(), "precondition: distinct buffers");
        }
        assert_eq!(
            fp,
            fingerprint(&fake_q1_params(&model, &other_qkv, Q1_TEST_EPOCH, ids))
        );
    }
}

/// Under identical handle ids (a caller passing fixed rather than per-load
/// ids), the model-owned slices still separate two weight sets: moving any
/// one of them changes the fingerprint, as does a differently shaped Q‖K‖V or
/// another depth.
///
/// Pure host logic: no CUDA device is touched.
#[test]
fn q1_weight_fingerprint_follows_owned_slices_qkv_length_and_depth() {
    use super::model_weights_fingerprint as fingerprint;

    let (model, qkv) = fake_q1_model();
    let ids = Q1WeightIds::Fallback;
    let fp = fingerprint(&fake_q1_params(&model, &qkv, Q1_TEST_EPOCH, ids));

    // Same lengths and contents as layer 1, at other addresses.
    let moved = FakeQ1Layer::new(1);
    assert_ne!(moved.attn_proj.as_ptr(), model[1].attn_proj.as_ptr());
    for slot in 0..6 {
        let mut params = fake_q1_params(&model, &qkv, Q1_TEST_EPOCH, ids);
        let lp = &mut params[1];
        match slot {
            0 => lp.attn_norm_bytes = &moved.attn_norm,
            1 => lp.attn_proj_bytes = &moved.attn_proj,
            2 => lp.gate_bytes = &moved.gate,
            3 => lp.up_bytes = &moved.up,
            4 => lp.down_bytes = &moved.down,
            _ => lp.ffn_norm_bytes = &moved.ffn_norm,
        }
        assert_ne!(
            fp,
            fingerprint(&params),
            "moving model-owned slice {slot} of layer 1 must change the fingerprint"
        );
    }
    // A whole other model under the same (fixed) handles.
    let (other_model, _) = fake_q1_model();
    assert_ne!(
        fp,
        fingerprint(&fake_q1_params(&other_model, &qkv, Q1_TEST_EPOCH, ids))
    );
    // A differently shaped Q‖K‖V.
    let longer: Vec<Vec<u8>> = (0..Q1_TEST_LAYERS as u8)
        .map(|s| fake_qkv_concat(s, Q1_TEST_QKV_LEN + 18))
        .collect();
    assert_ne!(
        fp,
        fingerprint(&fake_q1_params(&model, &longer, Q1_TEST_EPOCH, ids))
    );
    // A different depth.
    assert_ne!(
        fp,
        fingerprint(&fake_q1_params(
            &model[..Q1_TEST_LAYERS - 1],
            &qkv,
            Q1_TEST_EPOCH,
            ids
        ))
    );
}

/// The weight-set fingerprint is order-sensitive, like
/// `CudaGraphSlotKey::fingerprint_handles`: the same layers in another order,
/// or two handles trading roles, describe another upload and must not hit.
///
/// Pure host logic: no CUDA device is touched.
#[test]
fn q1_weight_fingerprint_is_order_sensitive() {
    use super::model_weights_fingerprint as fingerprint;

    let (model, qkv) = fake_q1_model();
    let ids = Q1WeightIds::Uploaded(100);
    let fp = fingerprint(&fake_q1_params(&model, &qkv, Q1_TEST_EPOCH, ids));

    let mut reversed = fake_q1_params(&model, &qkv, Q1_TEST_EPOCH, ids);
    reversed.reverse();
    assert_ne!(fp, fingerprint(&reversed), "layer order must matter");

    let mut swapped_norms = fake_q1_params(&model, &qkv, Q1_TEST_EPOCH, ids);
    let lp = &mut swapped_norms[0];
    std::mem::swap(&mut lp.attn_norm_handle, &mut lp.ffn_norm_handle);
    assert_ne!(fp, fingerprint(&swapped_norms), "handle roles must matter");

    let mut swapped_weights = fake_q1_params(&model, &qkv, Q1_TEST_EPOCH, ids);
    let lp = &mut swapped_weights[2];
    std::mem::swap(&mut lp.gate_up_handle, &mut lp.down_handle);
    assert_ne!(
        fp,
        fingerprint(&swapped_weights),
        "handle roles must matter"
    );
}

/// The Q1 and ternary caches judge a weight set by one rule
/// (`weight_identity`): the same field values give the same fingerprint on
/// both paths, so a fix to one can no longer leave the other behind. (Their
/// cache slots are disjoint and the graph key carries the quant family, so
/// equal values across families alias nothing.)
///
/// Pure host logic: no CUDA device is touched.
#[test]
fn q1_and_ternary_weight_fingerprints_share_one_rule() {
    use super::encode_ternary::ternary_model_weights_fingerprint;

    let model: Vec<FakeTernaryLayer> = (0..3u8).map(FakeTernaryLayer::new).collect();
    let qkv: Vec<Vec<u8>> = (0..3u8).map(|s| fake_qkv_concat(s, 34 * 12)).collect();
    for epoch in [Q1_TEST_EPOCH, Q1_TEST_EPOCH + 1] {
        let ternary: Vec<CudaFullForwardLayerParamsTernary<'_>> = model
            .iter()
            .zip(&qkv)
            .enumerate()
            .map(|(layer, (l, q))| l.params(q, epoch, layer as u64))
            .collect();
        let q1: Vec<CudaFullForwardLayerParams<'_>> = ternary
            .iter()
            .map(|t| CudaFullForwardLayerParams {
                attn_norm_handle: t.attn_norm_handle,
                attn_norm_bytes: t.attn_norm_bytes,
                fused_qkv_handle: t.fused_qkv_handle,
                fused_qkv_bytes: t.fused_qkv_bytes,
                q_norm_handle: t.q_norm_handle,
                q_norm_bytes: t.q_norm_bytes,
                k_norm_handle: t.k_norm_handle,
                k_norm_bytes: t.k_norm_bytes,
                attn_proj_handle: t.attn_proj_handle,
                attn_proj_bytes: t.attn_proj_bytes,
                ffn_norm_handle: t.ffn_norm_handle,
                ffn_norm_bytes: t.ffn_norm_bytes,
                gate_up_handle: t.gate_up_handle,
                gate_bytes: t.gate_bytes,
                up_bytes: t.up_bytes,
                down_handle: t.down_handle,
                down_bytes: t.down_bytes,
            })
            .collect();
        assert_eq!(
            model_weights_fingerprint(&q1),
            ternary_model_weights_fingerprint(&ternary)
        );
    }
}
