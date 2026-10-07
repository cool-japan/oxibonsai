//! Q1 weight-cache identity on real hardware: a second Q1 weight set of the
//! same depth, written into the **very host memory** a first one was decoded
//! from, must be decoded with its own weights.
//!
//! # Why
//!
//! The Q1 full forward (`try_cuda_full_forward`) caches one uploaded weight
//! set per process (`cuda_full_layer`'s `cached_q1_model_weights`) and reuses
//! it while the caller's layer parameters keep the same fingerprint. That
//! fingerprint used to hash the host slices' addresses and lengths only, so a
//! same-depth model whose host buffers landed on every address of a dropped
//! one matched the stale entry and was decoded with the dropped model's device
//! weights (the docs carried this as "one Q1 model per process"). It now also
//! mixes the handle ids, which `oxibonsai-model` composes over each model's
//! own `cuda_model_epoch` (`SlotNamespace`), as the ternary cache's
//! fingerprint already did.
//!
//! A model-level reload cannot force that address reuse (the allocator and
//! the GGUF layout decide where a reloaded model's tensors and its Q‖K‖V
//! concatenations land), so this drives the public kernel entry point with
//! caller-owned buffers instead, which makes the reuse exact:
//!
//! 1. References: weight sets A and B, each in its own buffer set (all alive
//!    at once, so at distinct addresses), each under a freshly minted epoch:
//!    `ref_a`, `ref_b`. Distinct addresses miss the cache under either
//!    fingerprint, so both are fresh uploads.
//! 2. Load X: weight set A written into a third buffer set: `out_x`.
//! 3. Load Y: weight set B written **in place** into that same buffer set
//!    (every slice keeps its address and length) under a fresh epoch, exactly
//!    what a reload sees when the allocator hands back every freed address:
//!    `out_y`.
//!
//! Oracles: `ref_a` and `ref_b` differ (the two sets are distinguishable);
//! `out_x == ref_a` (the comparison is deterministic across buffer sets and
//! handle ids); `out_y == ref_b` and `out_y != ref_a`. With the old
//! address-only fingerprint, load Y hits load X's cached weight set and
//! `out_y` equals `ref_a`.
//!
//! The weights are synthetic `Q1_0_g128` blocks (seeded signs, power-of-two
//! scales) at a scaled-down Qwen3 geometry with the real head shape (head_dim
//! 128, grouped KV heads), decoded at position 0.
//!
//! Self-skips (recording `cuda-hardware`, `executed: false`) when no CUDA
//! device answers `CudaGraph::global()`; `executed: true` is recorded only
//! after every oracle passed. Run with:
//!
//! ```text
//! cargo test -p oxibonsai-kernels --features native-cuda \
//!     --test cuda_q1_weight_cache_reload -- --test-threads=1 --nocapture
//! ```

#![cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]

use std::time::Instant;

use oxibonsai_kernels::gpu_backend::cuda_graph_slot::next_cuda_model_epoch;
use oxibonsai_kernels::{try_cuda_full_forward, CudaFullForwardLayerParams, CudaGraph};
use oxibonsai_testkit::capability::{record_executed_timed, record_skipped, Capability};

const TEST: &str = "oxibonsai-kernels::cuda_q1_weight_cache_reload::\
                    cuda_q1_weight_cache_reload_at_reused_host_addresses";

const N_LAYERS: usize = 2;
const HIDDEN: usize = 1024;
const N_Q_HEADS: usize = 8;
const N_KV_HEADS: usize = 4;
const HEAD_DIM: usize = 128;
const INTERMEDIATE: usize = 2048;
const MAX_SEQ: usize = 256;
const NORM_EPS: f32 = 1e-6;

/// `Q1_0_g128`: one FP16 scale, then 128 sign bits.
const Q1_BLOCK_BYTES: usize = 18;
const Q1_GROUP: usize = 128;

/// `SlotNamespace`'s composition (`oxibonsai-model`): `TAG | epoch << 24 |
/// local`, with the Q1 norm and weight-fallback locals.
const SLOT_TAG: u64 = 1 << 63;
const SLOT_LOCAL_BITS: u32 = 24;
const SLOT_EPOCH_MASK: u64 = (1 << (63 - SLOT_LOCAL_BITS)) - 1;
const NORM_LOCAL_BASE: u64 = 1_000_000;
const WEIGHT_FALLBACK_LOCAL_BASE: u64 = 4_000_000;

fn slot(epoch: u64, local: u64) -> u64 {
    SLOT_TAG | ((epoch & SLOT_EPOCH_MASK) << SLOT_LOCAL_BITS) | local
}

/// SplitMix64: deterministic test data without an RNG dependency.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `[lo, hi)`.
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        let unit = (self.next() >> 40) as f32 / (1u64 << 24) as f32;
        lo + (hi - lo) * unit
    }
}

/// Bytes of a `rows x cols` `Q1_0_g128` matrix.
const fn q1_bytes(rows: usize, cols: usize) -> usize {
    rows * (cols / Q1_GROUP) * Q1_BLOCK_BYTES
}

/// Overwrite `buf` in place with seeded `Q1_0_g128` blocks: a power-of-two
/// FP16 scale (2^-7, 2^-6 or 2^-5) and random sign bits per block.
fn fill_q1(buf: &mut [u8], rng: &mut SplitMix64) {
    const SCALES: [u16; 3] = [0x2000, 0x2400, 0x2800];
    let (blocks, tail) = buf.as_chunks_mut::<Q1_BLOCK_BYTES>();
    assert!(tail.is_empty(), "whole Q1_0_g128 blocks only");
    for block in blocks {
        let scale = SCALES[(rng.next() % 3) as usize];
        block[..2].copy_from_slice(&scale.to_le_bytes());
        let (sign_words, _) = block[2..].as_chunks_mut::<8>();
        for word in sign_words {
            *word = rng.next().to_le_bytes();
        }
    }
}

/// Overwrite `buf` in place with seeded RMSNorm weights in `[0.5, 1.5)`.
fn fill_norm(buf: &mut [f32], rng: &mut SplitMix64) {
    for v in buf.iter_mut() {
        *v = rng.uniform(0.5, 1.5);
    }
}

/// One layer's host buffers, allocated once and only ever overwritten in
/// place, so every slice keeps its address across weight sets.
struct LayerBuffers {
    attn_norm: Vec<f32>,
    fused_qkv: Vec<u8>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    attn_proj: Vec<u8>,
    ffn_norm: Vec<f32>,
    gate: Vec<u8>,
    up: Vec<u8>,
    down: Vec<u8>,
}

impl LayerBuffers {
    fn new() -> Self {
        let qkv_rows = (N_Q_HEADS + 2 * N_KV_HEADS) * HEAD_DIM;
        Self {
            attn_norm: vec![0.0; HIDDEN],
            fused_qkv: vec![0; q1_bytes(qkv_rows, HIDDEN)],
            q_norm: vec![0.0; HEAD_DIM],
            k_norm: vec![0.0; HEAD_DIM],
            attn_proj: vec![0; q1_bytes(HIDDEN, N_Q_HEADS * HEAD_DIM)],
            ffn_norm: vec![0.0; HIDDEN],
            gate: vec![0; q1_bytes(INTERMEDIATE, HIDDEN)],
            up: vec![0; q1_bytes(INTERMEDIATE, HIDDEN)],
            down: vec![0; q1_bytes(HIDDEN, INTERMEDIATE)],
        }
    }

    fn write(&mut self, rng: &mut SplitMix64) {
        fill_norm(&mut self.attn_norm, rng);
        fill_q1(&mut self.fused_qkv, rng);
        fill_norm(&mut self.q_norm, rng);
        fill_norm(&mut self.k_norm, rng);
        fill_q1(&mut self.attn_proj, rng);
        fill_norm(&mut self.ffn_norm, rng);
        fill_q1(&mut self.gate, rng);
        fill_q1(&mut self.up, rng);
        fill_q1(&mut self.down, rng);
    }

    /// `(address, length)` of every slice.
    fn identity(&self) -> [(usize, usize); 9] {
        fn id<T>(s: &[T]) -> (usize, usize) {
            (s.as_ptr() as usize, s.len())
        }
        [
            id(&self.attn_norm),
            id(&self.fused_qkv),
            id(&self.q_norm),
            id(&self.k_norm),
            id(&self.attn_proj),
            id(&self.ffn_norm),
            id(&self.gate),
            id(&self.up),
            id(&self.down),
        ]
    }
}

/// One model's worth of host buffers.
struct HostModel {
    layers: Vec<LayerBuffers>,
}

impl HostModel {
    fn new() -> Self {
        Self {
            layers: (0..N_LAYERS).map(|_| LayerBuffers::new()).collect(),
        }
    }

    /// Overwrite every buffer in place with weight set `seed`.
    fn write(&mut self, seed: u64) {
        let mut rng = SplitMix64(seed);
        for layer in &mut self.layers {
            layer.write(&mut rng);
        }
    }

    fn identity(&self) -> Vec<[(usize, usize); 9]> {
        self.layers.iter().map(LayerBuffers::identity).collect()
    }

    /// The layer parameters a model loaded under `epoch` hands the kernels,
    /// laid out like `oxibonsai-model`'s `build_cuda_layer_params` for blocks
    /// that were not uploaded at load (the weight-fallback slots).
    fn params(&self, epoch: u64) -> Vec<CudaFullForwardLayerParams<'_>> {
        self.layers
            .iter()
            .enumerate()
            .map(|(layer, b)| {
                let norm = slot(epoch, NORM_LOCAL_BASE + layer as u64 * 10);
                let weight = slot(epoch, WEIGHT_FALLBACK_LOCAL_BASE + layer as u64 * 4);
                CudaFullForwardLayerParams {
                    attn_norm_handle: norm,
                    attn_norm_bytes: &b.attn_norm,
                    fused_qkv_handle: weight,
                    fused_qkv_bytes: &b.fused_qkv,
                    q_norm_handle: norm + 1,
                    q_norm_bytes: &b.q_norm,
                    k_norm_handle: norm + 2,
                    k_norm_bytes: &b.k_norm,
                    attn_proj_handle: weight + 1,
                    attn_proj_bytes: &b.attn_proj,
                    ffn_norm_handle: norm + 3,
                    ffn_norm_bytes: &b.ffn_norm,
                    gate_up_handle: weight + 2,
                    gate_bytes: &b.gate,
                    up_bytes: &b.up,
                    down_handle: weight + 3,
                    down_bytes: &b.down,
                }
            })
            .collect()
    }
}

/// Layers-only Q1 full forward at position 0 for a model loaded under a
/// freshly minted epoch (a new load, as `BonsaiModel` construction mints one).
fn forward_as_new_load(model: &HostModel, hidden: &[f32], label: &str) -> Vec<f32> {
    let epoch = next_cuda_model_epoch();
    let params = model.params(epoch);
    let rope_cos = vec![1.0f32; HEAD_DIM / 2];
    let rope_sin = vec![0.0f32; HEAD_DIM / 2];
    let out = try_cuda_full_forward(
        hidden,
        &params,
        &rope_cos,
        &rope_sin,
        0,
        N_Q_HEADS,
        N_KV_HEADS,
        HEAD_DIM,
        N_Q_HEADS / N_KV_HEADS,
        NORM_EPS,
        HIDDEN,
        INTERMEDIATE,
        MAX_SEQ,
        None,
        0,
    );
    let Some(out) = out else {
        panic!("{label}: try_cuda_full_forward returned None on a live CUDA device");
    };
    assert_eq!(out.len(), HIDDEN, "{label}: output length");
    assert!(
        out.iter().all(|v| v.is_finite()),
        "{label}: non-finite output"
    );
    out
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn max_abs(a: &[f32]) -> f32 {
    a.iter().map(|v| v.abs()).fold(0.0f32, f32::max)
}

#[test]
fn cuda_q1_weight_cache_reload_at_reused_host_addresses() {
    let start = Instant::now();
    if let Err(e) = CudaGraph::global() {
        eprintln!("{TEST}: no CUDA device ({e}); skipping");
        record_skipped(Capability::CudaHardware, TEST);
        return;
    }

    const SEED_A: u64 = 0xa11c_e5ee_d000_0001;
    const SEED_B: u64 = 0xb0b5_eed0_0000_0002;
    let mut input_rng = SplitMix64(0x1234_5678_9abc_def0);
    let hidden: Vec<f32> = (0..HIDDEN).map(|_| input_rng.uniform(-1.0, 1.0)).collect();

    // All three buffer sets stay alive to the end, so they never share an
    // address with each other.
    let mut model_a = HostModel::new();
    let mut model_b = HostModel::new();
    let mut reused = HostModel::new();
    model_a.write(SEED_A);
    model_b.write(SEED_B);
    let (ia, ib, ir) = (model_a.identity(), model_b.identity(), reused.identity());
    assert!(
        ia[0][1].0 != ib[0][1].0 && ia[0][1].0 != ir[0][1].0 && ib[0][1].0 != ir[0][1].0,
        "precondition: three distinct buffer sets"
    );

    // 1. References, each a fresh upload at its own addresses.
    let ref_a = forward_as_new_load(&model_a, &hidden, "reference A");
    let ref_b = forward_as_new_load(&model_b, &hidden, "reference B");

    // 2. Load X: A in the reused buffer set.
    reused.write(SEED_A);
    let out_x = forward_as_new_load(&reused, &hidden, "load X (A)");

    // 3. Load Y: B written in place into the same buffer set — every slice at
    //    the address and length load X used — under a new epoch.
    reused.write(SEED_B);
    assert_eq!(
        reused.identity(),
        ir,
        "precondition: load Y reuses every host address of load X"
    );
    let out_y = forward_as_new_load(&reused, &hidden, "load Y (B, reused addresses)");

    let scale = max_abs(&ref_a).max(max_abs(&ref_b)).max(1.0);
    let d_ab = max_abs_diff(&ref_a, &ref_b);
    let d_xa = max_abs_diff(&out_x, &ref_a);
    let d_yb = max_abs_diff(&out_y, &ref_b);
    let d_ya = max_abs_diff(&out_y, &ref_a);
    println!(
        "{TEST}: max|ref_a|={:.4} max|ref_b|={:.4}; max|ref_a-ref_b|={d_ab:.6e} \
         max|X-ref_a|={d_xa:.6e} max|Y-ref_b|={d_yb:.6e} max|Y-ref_a|={d_ya:.6e}",
        max_abs(&ref_a),
        max_abs(&ref_b),
    );

    let same = 1e-5 * scale;
    let distinct = 1e-2 * scale;
    assert!(
        d_ab > distinct,
        "weight sets A and B must be distinguishable (max|ref_a-ref_b| = {d_ab:e})"
    );
    assert!(
        d_xa <= same,
        "load X must reproduce reference A (max|X-ref_a| = {d_xa:e}): the comparison is not \
         deterministic across buffer sets and handle ids"
    );
    assert!(
        d_ya > distinct,
        "load Y decoded with weight set A's cached device weights (max|Y-ref_a| = {d_ya:e}): \
         a same-depth Q1 reload at reused host addresses hit the previous weight set"
    );
    assert!(
        d_yb <= same,
        "load Y must reproduce reference B (max|Y-ref_b| = {d_yb:e})"
    );
    println!("{TEST}: PASS (load Y decoded with its own weights at reused host addresses)");
    record_executed_timed(Capability::CudaHardware, TEST, start.elapsed());
}
