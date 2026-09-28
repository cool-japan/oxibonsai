//! Integration tests for the GPU backend abstraction layer.
//!
//! All tests use `CpuBackend` — no real GPU hardware is required.

use oxibonsai_kernels::gpu_backend::{
    gpu_gemv_1bit, gpu_matmul, select_backend, CpuBackend, DeviceBuffer, GpuBackendTrait,
    LaunchConfig,
};

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn backend() -> CpuBackend {
    CpuBackend::new()
}

// ---------------------------------------------------------------------------
// 1. name()
// ---------------------------------------------------------------------------
#[test]
fn cpu_backend_name() {
    let b = backend();
    assert_eq!(b.name(), "cpu");
}

// ---------------------------------------------------------------------------
// 2. is_accelerated()
// ---------------------------------------------------------------------------
#[test]
fn cpu_backend_not_accelerated() {
    let b = backend();
    assert!(!b.is_accelerated());
}

// ---------------------------------------------------------------------------
// 3. device_count()
// ---------------------------------------------------------------------------
#[test]
fn cpu_backend_device_count() {
    let b = backend();
    assert!(b.device_count() >= 1);
}

// ---------------------------------------------------------------------------
// 4. alloc()
// ---------------------------------------------------------------------------
#[test]
fn cpu_backend_alloc() {
    let b = backend();
    let buf = b.alloc(100, 0).expect("alloc should succeed");
    assert_eq!(buf.size(), 100);
    assert_eq!(buf.device_id(), 0);
    // Freshly allocated buffer must be zero-filled.
    assert!(buf.data.iter().all(|&v| v == 0.0_f32));
}

// ---------------------------------------------------------------------------
// 5. host_to_device()
// ---------------------------------------------------------------------------
#[test]
fn cpu_backend_host_to_device() {
    let b = backend();
    let src: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let buf = b
        .host_to_device(&src, 0)
        .expect("host_to_device should succeed");
    assert_eq!(buf.size(), 8);
    assert_eq!(buf.to_vec(), src);
}

// ---------------------------------------------------------------------------
// 6. device_to_host() round-trip
// ---------------------------------------------------------------------------
#[test]
fn cpu_backend_device_to_host() {
    let b = backend();
    let src = vec![1.0_f32, 2.0, 3.0, 4.0];
    let buf = b.host_to_device(&src, 0).expect("host_to_device");
    let out = b.device_to_host(&buf).expect("device_to_host");
    assert_eq!(out, src);
}

// ---------------------------------------------------------------------------
// 7. matvec 2×2 identity check
// ---------------------------------------------------------------------------
#[test]
fn cpu_backend_matvec_2x2() {
    let b = backend();
    // A = [[1,0],[0,1]], x = [3,4]  →  y = [3,4]
    let a = b.host_to_device(&[1.0, 0.0, 0.0, 1.0], 0).expect("h2d a");
    let x = b.host_to_device(&[3.0, 4.0], 0).expect("h2d x");
    let y = b.matvec(&a, &x, 2, 2, 0).expect("matvec");
    let result = b.device_to_host(&y).expect("d2h");
    assert!((result[0] - 3.0).abs() < 1e-6, "y[0] = {}", result[0]);
    assert!((result[1] - 4.0).abs() < 1e-6, "y[1] = {}", result[1]);
}

// ---------------------------------------------------------------------------
// 8. matvec with explicit identity — general case
// ---------------------------------------------------------------------------
#[test]
fn cpu_backend_matvec_identity() {
    let b = backend();
    let n = 4_usize;
    // Build n×n identity matrix (row-major)
    let mut identity = vec![0.0_f32; n * n];
    for i in 0..n {
        identity[i * n + i] = 1.0;
    }
    let x_data: Vec<f32> = (1..=n as u32).map(|v| v as f32).collect();

    let a_buf = b.host_to_device(&identity, 0).expect("h2d a");
    let x_buf = b.host_to_device(&x_data, 0).expect("h2d x");
    let y_buf = b.matvec(&a_buf, &x_buf, n, n, 0).expect("matvec");
    let y = b.device_to_host(&y_buf).expect("d2h");

    for i in 0..n {
        assert!(
            (y[i] - x_data[i]).abs() < 1e-5,
            "y[{}] = {} != {}",
            i,
            y[i],
            x_data[i]
        );
    }
}

// ---------------------------------------------------------------------------
// 9. relu with mixed signs
// ---------------------------------------------------------------------------
#[test]
fn cpu_backend_relu_positive() {
    let b = backend();
    let input = vec![1.0_f32, -1.0, 2.0];
    let buf = b.host_to_device(&input, 0).expect("h2d");
    let out_buf = b.relu(&buf, 0).expect("relu");
    let out = b.device_to_host(&out_buf).expect("d2h");
    assert!((out[0] - 1.0).abs() < 1e-6);
    assert!((out[1] - 0.0).abs() < 1e-6);
    assert!((out[2] - 2.0).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// 10. relu all-negative
// ---------------------------------------------------------------------------
#[test]
fn cpu_backend_relu_all_negative() {
    let b = backend();
    let input = vec![-1.0_f32, -2.0];
    let buf = b.host_to_device(&input, 0).expect("h2d");
    let out_buf = b.relu(&buf, 0).expect("relu");
    let out = b.device_to_host(&out_buf).expect("d2h");
    for &v in &out {
        assert!((v - 0.0).abs() < 1e-6, "expected 0, got {v}");
    }
}

// ---------------------------------------------------------------------------
// 11. softmax sums to 1
// ---------------------------------------------------------------------------
#[test]
fn cpu_backend_softmax_sums_to_one() {
    let b = backend();
    let input = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let size = input.len();
    let buf = b.host_to_device(&input, 0).expect("h2d");
    let out_buf = b.softmax(&buf, size, 0).expect("softmax");
    let out = b.device_to_host(&out_buf).expect("d2h");
    let sum: f32 = out.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "softmax sum = {sum}, expected ~1.0"
    );
}

// ---------------------------------------------------------------------------
// 12. softmax uniform input → uniform output
// ---------------------------------------------------------------------------
#[test]
fn cpu_backend_softmax_uniform() {
    let b = backend();
    let input = vec![0.0_f32; 3];
    let size = input.len();
    let buf = b.host_to_device(&input, 0).expect("h2d");
    let out_buf = b.softmax(&buf, size, 0).expect("softmax");
    let out = b.device_to_host(&out_buf).expect("d2h");
    let expected = 1.0 / 3.0_f32;
    for &v in &out {
        assert!((v - expected).abs() < 1e-5, "expected ~{expected}, got {v}");
    }
}

// ---------------------------------------------------------------------------
// 13. synchronize returns Ok
// ---------------------------------------------------------------------------
#[test]
fn cpu_backend_synchronize_ok() {
    let b = backend();
    b.synchronize(0).expect("synchronize should succeed");
}

// ---------------------------------------------------------------------------
// 14. memory_info: free <= total
// ---------------------------------------------------------------------------
#[test]
fn cpu_backend_memory_info() {
    let b = backend();
    let (free, total) = b.memory_info(0).expect("memory_info should succeed");
    assert!(free <= total, "free ({free}) must be <= total ({total})");
    assert!(total > 0, "total memory must be > 0");
}

// ---------------------------------------------------------------------------
// 15. select_backend returns a usable backend
// ---------------------------------------------------------------------------
#[test]
fn select_backend_returns_usable() {
    let b = select_backend();
    // Must have a non-empty name regardless of backend.
    assert!(!b.name().is_empty());
    // On machines with Metal/CUDA, is_accelerated() may be true.
    // Without gpu features, stubs are never accelerated.
    #[cfg(not(feature = "gpu"))]
    assert!(!b.is_accelerated());
}

// ---------------------------------------------------------------------------
// 16. gpu_matmul with identity matrix
// ---------------------------------------------------------------------------
#[test]
fn gpu_matmul_identity() {
    let b = backend();
    let n = 3_usize;
    // Identity matrix I_n
    let mut identity = vec![0.0_f32; n * n];
    for i in 0..n {
        identity[i * n + i] = 1.0;
    }
    // B is a 3×2 matrix
    let b_mat = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let result = gpu_matmul(&b, &identity, &b_mat, n, n, 2, 0).expect("gpu_matmul");
    // I * B == B
    for (r, expected) in result.iter().zip(b_mat.iter()) {
        assert!((r - expected).abs() < 1e-5, "got {r}, expected {expected}");
    }
}

// ---------------------------------------------------------------------------
// 17. LaunchConfig::for_n_elements produces reasonable dimensions
// ---------------------------------------------------------------------------
#[test]
fn launch_config_for_n() {
    let cfg = LaunchConfig::for_n_elements(1024);
    // block_dim.0 should be 256 by default
    assert_eq!(cfg.block_dim.0, 256);
    // grid must cover 1024 elements: ceil(1024/256) = 4
    assert_eq!(cfg.grid_dim.0, 4);
    assert_eq!(cfg.grid_dim.1, 1);
    assert_eq!(cfg.grid_dim.2, 1);
    assert_eq!(cfg.block_dim.1, 1);
    assert_eq!(cfg.block_dim.2, 1);
}

// ---------------------------------------------------------------------------
// 18. DeviceBuffer::size() matches allocated size
// ---------------------------------------------------------------------------
#[test]
fn device_buffer_size() {
    let sizes = [0_usize, 1, 64, 1024, 65536];
    for &s in &sizes {
        let buf = DeviceBuffer::new(s, 0);
        assert_eq!(buf.size(), s, "DeviceBuffer::size() mismatch for size={s}");
    }
}

// ===========================================================================
// Q1_0_g128 GPU GEMV tests (CPU fallback path)
// ===========================================================================

/// Serializes every test below that calls [`gpu_gemv_1bit`].
///
/// Wave-1.5 addendum (3): under `--all-features` (or `--features metal`),
/// `gpu_gemv_1bit` routes through `Scirs2Backend::global()`, which in turn
/// resolves a Metal session. MET-08 (METAL-CONCURRENCY, landed wave 4) split
/// the old single mutable `GLOBAL_METAL_GRAPH` into a process-shared,
/// immutable `MetalDevice` and a per-session `MetalGraph` bound to a thread
/// via `SessionScope` — but none of the tests below bind one, so
/// `MetalGraph::global()` still falls through to the single
/// process-default session (`session.rs`'s documented fallback for a caller
/// that never binds), the same shared mutable GPU state this guard was
/// written against. `cargo test`'s default in-binary thread parallelism let
/// two of these tests race and intermittently observe each other's
/// half-finished GPU state (symptom: "got 0 expected 8128", i.e. a
/// GEMV result read before the corresponding command buffer completed).
/// `cargo nextest` hides this (one process per test), and a plain
/// `--test-threads=1` run doesn't reproduce it either, but the *default*
/// `cargo test` invocation is real and part of this project's gate. The
/// precedent for this exact fix is
/// `crates/oxibonsai-model/tests/metal_prefill_ternary_parity_tests.rs::gpu_serial`
/// (FIX-06-KERN-MODEL); rewriting these tests to bind their own
/// `MetalGraph::bind_new_session()` would let this helper (and every
/// `let _gpu = gpu_serial();` call site below) go, but that rewrite is out
/// of this package's scope.
fn gpu_serial() -> std::sync::MutexGuard<'static, ()> {
    static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Helper: build a single BlockQ1_0G128 as raw bytes (18 bytes).
fn make_q1_block(scale: f32, bits: [u8; 16]) -> Vec<u8> {
    let d = half::f16::from_f32(scale);
    let mut block = Vec::with_capacity(18);
    block.extend_from_slice(&d.to_bits().to_le_bytes());
    block.extend_from_slice(&bits);
    block
}

// ---------------------------------------------------------------------------
// 19. gpu_gemv_1bit: all-ones bits, scale=1 → sum of input
// ---------------------------------------------------------------------------
#[test]
fn gemv_1bit_all_ones_scale_one() {
    let _gpu = gpu_serial();
    let block = make_q1_block(1.0, [0xFF; 16]);
    let input: Vec<f32> = (0..128).map(|i| i as f32).collect();
    let expected: f32 = input.iter().sum(); // 8128
    let result = gpu_gemv_1bit(&block, &input, 1, 128).expect("gemv_1bit");
    assert!(
        (result[0] - expected).abs() < 1.0,
        "got {} expected {}",
        result[0],
        expected,
    );
}

// ---------------------------------------------------------------------------
// 20. gpu_gemv_1bit: all-zeros bits → neg sum
// ---------------------------------------------------------------------------
#[test]
fn gemv_1bit_all_zeros_neg_sum() {
    let _gpu = gpu_serial();
    let block = make_q1_block(1.0, [0x00; 16]);
    let input = vec![1.0_f32; 128];
    // all bits=0 → weight = -1 → output = -128
    let result = gpu_gemv_1bit(&block, &input, 1, 128).expect("gemv_1bit");
    assert!(
        (result[0] - (-128.0)).abs() < 1.0,
        "got {} expected -128",
        result[0],
    );
}

// ---------------------------------------------------------------------------
// 21. gpu_gemv_1bit: multi-row
// ---------------------------------------------------------------------------
#[test]
fn gemv_1bit_multi_row() {
    let _gpu = gpu_serial();
    // 2 rows × k=128
    let row0 = make_q1_block(2.0, [0xFF; 16]); // +2 * input
    let row1 = make_q1_block(0.5, [0x00; 16]); // -0.5 * input
    let mut blocks = row0;
    blocks.extend_from_slice(&row1);

    let input = vec![1.0_f32; 128];
    let result = gpu_gemv_1bit(&blocks, &input, 2, 128).expect("gemv_1bit 2 rows");
    // row0: 2.0 * 128 * 1.0 = 256
    assert!((result[0] - 256.0).abs() < 1.0, "row0: got {}", result[0]);
    // row1: -0.5 * 128 * 1.0 = -64
    assert!((result[1] - (-64.0)).abs() < 1.0, "row1: got {}", result[1]);
}

// ---------------------------------------------------------------------------
// 22. gpu_gemv_1bit: k=256 (two blocks per row)
// ---------------------------------------------------------------------------
#[test]
fn gemv_1bit_two_blocks_per_row() {
    let _gpu = gpu_serial();
    let b0 = make_q1_block(1.0, [0xFF; 16]); // bits=1 → +1
    let b1 = make_q1_block(1.0, [0x00; 16]); // bits=0 → -1
    let mut blocks = b0;
    blocks.extend_from_slice(&b1);

    let input = vec![1.0_f32; 256];
    let result = gpu_gemv_1bit(&blocks, &input, 1, 256).expect("gemv_1bit k=256");
    // block0: +128, block1: -128 → 0
    assert!(result[0].abs() < 1.0, "expected ~0, got {}", result[0],);
}

// ---------------------------------------------------------------------------
// 23. gpu_gemv_1bit: error on bad k
// ---------------------------------------------------------------------------
#[test]
fn gemv_1bit_bad_k_not_multiple_128() {
    let _gpu = gpu_serial();
    let result = gpu_gemv_1bit(&[], &[], 0, 64);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// 24. gpu_gemv_1bit: error on mismatched input size
// ---------------------------------------------------------------------------
#[test]
fn gemv_1bit_input_size_mismatch() {
    let _gpu = gpu_serial();
    let block = make_q1_block(1.0, [0xFF; 16]);
    let input = vec![1.0_f32; 64]; // should be 128
    let result = gpu_gemv_1bit(&block, &input, 1, 128);
    assert!(result.is_err());
}
