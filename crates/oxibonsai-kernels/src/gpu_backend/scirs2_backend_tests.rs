//! Unit tests for [`super::Scirs2Backend`].
//!
//! Split out of `scirs2_backend.rs` (which was at 1955 of the 2000-line policy
//! ceiling) and re-attached with `#[cfg(test)] #[path]`, so these remain a
//! child module of `scirs2_backend` and keep reaching its private items via
//! `use super::*`.
//!
//! Every GPU-touching test early-returns when `is_accelerated()` is false, so
//! the suite is green on hosts with no usable GPU.

use super::*;

fn make_backend() -> Option<Scirs2Backend> {
    match Scirs2Backend::new() {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("Scirs2Backend not available: {e}");
            None
        }
    }
}

#[test]
fn scirs2_backend_creation() {
    let _backend = make_backend();
    // If GPU is not available, we just skip.
}

#[test]
fn scirs2_backend_name_is_not_empty() {
    if let Some(b) = make_backend() {
        assert!(!b.name().is_empty());
    }
}

#[test]
fn scirs2_backend_alloc() {
    if let Some(b) = make_backend() {
        let buf = b.alloc(64, 0).expect("alloc");
        assert_eq!(buf.size(), 64);
    }
}

#[test]
fn scirs2_backend_host_roundtrip() {
    if let Some(b) = make_backend() {
        let src = vec![1.0_f32, 2.0, 3.0, 4.0];
        let buf = b.host_to_device(&src, 0).expect("h2d");
        let out = b.device_to_host(&buf).expect("d2h");
        assert_eq!(out, src);
    }
}

// ── MET-M1: weight-cache opt-in + eviction ──────────────────────────

#[test]
fn weight_cache_enabled_defaults_to_true() {
    if let Some(b) = make_backend() {
        assert!(b.weight_cache_enabled());
    }
}

#[test]
fn with_weight_cache_enabled_builder_sets_flag() {
    if let Some(b) = make_backend() {
        let b = b.with_weight_cache_enabled(false);
        assert!(!b.weight_cache_enabled());
        let b = b.with_weight_cache_enabled(true);
        assert!(b.weight_cache_enabled());
    }
}

#[test]
fn set_weight_cache_enabled_toggles_on_shared_instance() {
    if let Some(b) = make_backend() {
        assert!(b.weight_cache_enabled());
        b.set_weight_cache_enabled(false);
        assert!(!b.weight_cache_enabled());
        b.set_weight_cache_enabled(true);
        assert!(b.weight_cache_enabled());
    }
}

#[test]
fn upload_weights_respects_disabled_cache() {
    if let Some(b) = make_backend() {
        if !b.is_accelerated() {
            return;
        }
        b.set_weight_cache_enabled(false);
        let handle = b
            .upload_weights(&[0u8; 18])
            .expect("upload should still succeed when caching is disabled");
        // The upload happened, but nothing was retained.
        assert_eq!(b.cached_weight_count(), 0);
        // ...so a later cached lookup using this handle must fail, not
        // silently resolve to someone else's buffer.
        assert!(b
            .gemv_q1_g128_cached(handle, &[0.0f32; 128], 1, 128)
            .is_err());
    }
}

#[test]
fn upload_weights_retains_when_cache_enabled() {
    if let Some(b) = make_backend() {
        if !b.is_accelerated() {
            return;
        }
        assert!(b.weight_cache_enabled(), "default should be enabled");
        let _handle = b.upload_weights(&[0u8; 18]).expect("upload should succeed");
        assert_eq!(b.cached_weight_count(), 1);
    }
}

#[test]
fn clear_weight_cache_evicts_all_entries() {
    if let Some(b) = make_backend() {
        if !b.is_accelerated() {
            return;
        }
        b.upload_weights(&[0u8; 18]).expect("upload 1");
        b.upload_weights(&[1u8; 18]).expect("upload 2");
        assert_eq!(b.cached_weight_count(), 2);
        let evicted = b.clear_weight_cache().expect("clear_weight_cache");
        assert_eq!(evicted, 2);
        assert_eq!(b.cached_weight_count(), 0);
    }
}

#[test]
fn clear_weight_cache_on_empty_cache_is_a_noop() {
    if let Some(b) = make_backend() {
        let evicted = b.clear_weight_cache().expect("clear_weight_cache");
        assert_eq!(evicted, 0);
    }
}

#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn tq2_upload_ternary_creates_handle() {
    use half::f16;
    use oxibonsai_core::BlockTQ2_0_g128;
    if let Some(b) = make_backend() {
        if !b.is_accelerated() {
            return;
        }
        let block = BlockTQ2_0_g128 {
            qs: [0xAAu8; 32],
            d: f16::from_f32(1.0),
        };
        let handle = b.upload_weights_ternary(&[block]);
        assert!(
            handle.is_ok(),
            "ternary upload should succeed: {:?}",
            handle
        );
        assert_eq!(b.cached_weight_count(), 1);
    }
}

#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn tq2_gemv_all_positive_weights() {
    use half::f16;
    use oxibonsai_core::BlockTQ2_0_g128;
    if let Some(b) = make_backend() {
        if !b.is_accelerated() {
            return;
        }
        // 1 row, k=128: qs=0xAA → all +1, scale=1.0, input all 1.0
        // Expected: 128 × 1.0 × 1.0 = 128.0
        let block = BlockTQ2_0_g128 {
            qs: [0xAAu8; 32],
            d: f16::from_f32(1.0),
        };
        let handle = b
            .upload_weights_ternary(&[block])
            .expect("upload should succeed");
        let input = vec![1.0f32; 128];
        let result = b.gemv_tq2_g128_cached(handle, &input, 1, 128);
        assert!(
            result.is_ok(),
            "cached ternary GEMV should succeed: {:?}",
            result
        );
        let out = result.expect("result");
        assert_eq!(out.len(), 1);
        assert!(
            (out[0] - 128.0).abs() < 1.0,
            "expected ~128.0, got {}",
            out[0]
        );
    }
}

// ── Concurrency: the shared I/O pool + shared kernel handles ─────────

/// Build a single `BlockQ1_0G128` (18 bytes: f16 scale, then 16 sign bytes).
fn q1_block(scale: f32, bits: [u8; 16]) -> Vec<u8> {
    let mut block = Vec::with_capacity(18);
    block.extend_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
    block.extend_from_slice(&bits);
    block
}

/// Distinct `(weights, input)` workload for thread `t`, chosen so any
/// cross-talk between concurrent callers produces a grossly wrong number
/// rather than a near-miss: the scale, the sign pattern and the input
/// magnitude all differ per thread.
fn race_workload(t: usize) -> (Vec<u8>, Vec<f32>) {
    // Vary the sign byte *within* the block as well as across threads: a
    // single repeated byte can make the row sum cancel to exactly zero
    // (0x66 does, for a linearly increasing input), which would make two
    // workloads indistinguishable and the test vacuous.
    let mut bits = [0u8; 16];
    for (j, b) in bits.iter_mut().enumerate() {
        *b = ((t * 31 + j * 17 + 5) % 251) as u8;
    }
    let weights = q1_block(0.5 + t as f32, bits);
    let input: Vec<f32> = (0..128)
        .map(|i| (i as f32 + 1.0) * (t as f32 + 1.0))
        .collect();
    (weights, input)
}

/// Join every worker and report **all** failures, not just the first: a
/// race that only hits one of N threads is exactly what this is for.
fn join_workers(handles: Vec<std::thread::JoinHandle<Result<(), String>>>) {
    let mut failures = Vec::new();
    for h in handles {
        match h.join() {
            Ok(Ok(())) => {}
            Ok(Err(msg)) => failures.push(msg),
            Err(_) => failures.push("worker thread panicked".to_string()),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Reproduces the `Scirs2Backend` shared-state data race.
///
/// `gemv_q1_g128` used to copy the caller's input into ONE process-global
/// `GpuBuffer<f32>`, release the mutex, and only then bind and dispatch on
/// a compiled kernel handle that is also process-global — so two
/// concurrent callers of the public `gpu_gemv_1bit` entry point could
/// overwrite each other's input, each other's kernel parameters, and read
/// each other's output. It is invisible to `cargo nextest` (one process
/// per test) and shows up under `cargo test`'s in-process parallelism.
///
/// Every thread recomputes the same workload many times and compares
/// against a reference captured single-threaded, so a single instance of
/// cross-talk fails the test. Bit-exact: identical bytes through an
/// identical kernel.
///
/// Skipped (early `return`, not `#[ignore]`) when this backend is not
/// hardware-accelerated — `gpu_gemv_1bit` then takes the pure-CPU
/// fallback, which has no shared state and could not fail this.
#[test]
fn concurrent_gpu_gemv_1bit_callers_do_not_corrupt_each_other() {
    use crate::gpu_backend::gpu_gemv_1bit;

    let backend = match Scirs2Backend::global() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("no GPU backend ({e}); skipping");
            return;
        }
    };
    if !backend.is_accelerated() {
        eprintln!("GPU backend is the CPU fallback; the shared-buffer path is not taken");
        return;
    }

    const THREADS: usize = 6;
    const ITERS: usize = 60;

    // Single-threaded references, one per workload.
    let expected: Vec<Vec<f32>> = (0..THREADS)
        .map(|t| {
            let (w, input) = race_workload(t);
            gpu_gemv_1bit(&w, &input, 1, 128).expect("reference gemv")
        })
        .collect();
    // The workloads must be distinguishable, or the test proves nothing.
    for t in 1..THREADS {
        assert!(
            (expected[t][0] - expected[0][0]).abs() > 1e-3,
            "degenerate fixture: workload {t} and 0 give the same result \
                 ({} vs {}), so cross-talk would be undetectable",
            expected[t][0],
            expected[0][0]
        );
    }

    let expected = std::sync::Arc::new(expected);
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(THREADS));
    let mut handles = Vec::with_capacity(THREADS);
    for t in 0..THREADS {
        let expected = std::sync::Arc::clone(&expected);
        let barrier = std::sync::Arc::clone(&barrier);
        handles.push(std::thread::spawn(move || -> Result<(), String> {
            let (w, input) = race_workload(t);
            barrier.wait();
            for iter in 0..ITERS {
                let got = gpu_gemv_1bit(&w, &input, 1, 128)
                    .map_err(|e| format!("thread {t} iter {iter}: {e}"))?;
                if got != expected[t] {
                    return Err(format!(
                        "thread {t} iter {iter}: got {got:?}, expected {:?} — a \
                             concurrent caller clobbered the shared input/output \
                             buffer or the shared kernel parameters",
                        expected[t]
                    ));
                }
            }
            Ok(())
        }));
    }
    join_workers(handles);
}

/// The batched sibling shares the very same pool and kernel handle, so it
/// gets the same guarantee — and mixing the two entry points concurrently
/// is the case that would catch a fix applied to only one of them.
#[test]
fn concurrent_gemv_and_gemm_q1_callers_do_not_corrupt_each_other() {
    let backend = match Scirs2Backend::global() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("no GPU backend ({e}); skipping");
            return;
        }
    };
    if !backend.is_accelerated() {
        eprintln!("GPU backend is the CPU fallback; the shared-buffer path is not taken");
        return;
    }

    const THREADS: usize = 6;
    const ITERS: usize = 40;
    const BATCH: usize = 3;

    let gemv_expected: Vec<Vec<f32>> = (0..THREADS)
        .map(|t| {
            let (w, input) = race_workload(t);
            backend
                .gemv_q1_g128(&w, &input, 1, 128)
                .expect("reference gemv")
        })
        .collect();
    let gemm_expected: Vec<Vec<f32>> = (0..THREADS)
        .map(|t| {
            let (w, input) = race_workload(t);
            let batched: Vec<f32> = (0..BATCH)
                .flat_map(|b| input.iter().map(move |v| v * (b as f32 + 1.0)))
                .collect();
            backend
                .gemm_q1_g128(&w, &batched, BATCH, 1, 128)
                .expect("reference gemm")
        })
        .collect();

    let barrier = std::sync::Arc::new(std::sync::Barrier::new(THREADS));
    let gemv_expected = std::sync::Arc::new(gemv_expected);
    let gemm_expected = std::sync::Arc::new(gemm_expected);
    let mut handles = Vec::with_capacity(THREADS);
    for t in 0..THREADS {
        let backend = std::sync::Arc::clone(&backend);
        let barrier = std::sync::Arc::clone(&barrier);
        let gemv_expected = std::sync::Arc::clone(&gemv_expected);
        let gemm_expected = std::sync::Arc::clone(&gemm_expected);
        handles.push(std::thread::spawn(move || -> Result<(), String> {
            let (w, input) = race_workload(t);
            let batched: Vec<f32> = (0..BATCH)
                .flat_map(|b| input.iter().map(move |v| v * (b as f32 + 1.0)))
                .collect();
            barrier.wait();
            for iter in 0..ITERS {
                // Alternate the two entry points so they interleave.
                if iter % 2 == 0 {
                    let got = backend
                        .gemv_q1_g128(&w, &input, 1, 128)
                        .map_err(|e| format!("thread {t} iter {iter} gemv: {e}"))?;
                    if got != gemv_expected[t] {
                        return Err(format!(
                            "thread {t} iter {iter}: gemv got {got:?}, expected {:?}",
                            gemv_expected[t]
                        ));
                    }
                } else {
                    let got = backend
                        .gemm_q1_g128(&w, &batched, BATCH, 1, 128)
                        .map_err(|e| format!("thread {t} iter {iter} gemm: {e}"))?;
                    if got != gemm_expected[t] {
                        return Err(format!(
                            "thread {t} iter {iter}: gemm got {got:?}, expected {:?}",
                            gemm_expected[t]
                        ));
                    }
                }
            }
            Ok(())
        }));
    }
    join_workers(handles);
}

// ── Q1 replica sharing + per-epoch release (MET-M1) ──

/// Two distinct 18-byte Q1 blocks, so "same bytes" and "different bytes"
/// cases are unambiguous.
fn block_a() -> Vec<u8> {
    q1_block(1.0, [0x5Au8; 16])
}

fn block_b() -> Vec<u8> {
    q1_block(2.0, [0xA5u8; 16])
}

/// Run `f` with this thread's uploads attributed to `epoch`, returning its
/// result and what the scope saw.
fn in_epoch<T>(epoch: u64, f: impl FnOnce() -> T) -> (T, crate::gpu_backend::UploadStats) {
    let scope = crate::gpu_backend::GpuUploadScope::enter(epoch);
    let out = f();
    (out, scope.finish())
}

#[test]
fn content_fingerprint_separates_content_and_length() {
    let a = content_fingerprint(&block_a());
    let b = content_fingerprint(&block_b());
    assert_ne!(a, b, "different content must fingerprint differently");
    assert_eq!(a, content_fingerprint(&block_a()), "deterministic");
    // A zero-padded tail must not collide with the shorter input.
    let short = [1u8, 2, 3];
    let padded = [1u8, 2, 3, 0];
    assert_ne!(content_fingerprint(&short), content_fingerprint(&padded));
    assert_ne!(content_fingerprint(&[]), content_fingerprint(&[0u8]));
}

/// The replica-sharing finding, at the backend: a second
/// upload of byte-identical weights from another replica (another epoch)
/// must return the SAME handle — which is also what makes the Q1 fused
/// path's `MetalGraph` slots coincide — and place nothing new on the device.
#[test]
fn identical_uploads_from_two_epochs_share_one_resident_buffer() {
    let Some(b) = make_backend() else { return };
    let epoch_a = crate::gpu_backend::next_gpu_model_epoch();
    let epoch_b = crate::gpu_backend::next_gpu_model_epoch();
    let bytes = block_a();

    let (handle_a, stats_a) = in_epoch(epoch_a, || b.upload_weights(&bytes));
    let (handle_b, stats_b) = in_epoch(epoch_b, || b.upload_weights(&bytes));
    let handle_a = handle_a.expect("upload A");
    let handle_b = handle_b.expect("upload B");

    assert_eq!(handle_a, handle_b, "identical bytes must share one handle");
    assert_eq!(b.cached_weight_count(), 1, "one resident buffer, not two");
    assert_eq!(b.cached_weight_bytes(), bytes.len() as u64);
    assert_eq!(
        b.uploaded_weight_bytes_total(),
        bytes.len() as u64,
        "bytes uploaded must grow once, not per replica"
    );
    assert_eq!(stats_a.fresh_buffers, 1);
    assert_eq!(stats_a.shared_buffers, 0);
    assert_eq!(stats_b.fresh_buffers, 0);
    assert_eq!(stats_b.shared_buffers, 1);
    assert_eq!(stats_b.shared_bytes, bytes.len() as u64);
    assert_eq!(b.epoch_registration_count(epoch_a), 1);
    assert_eq!(b.epoch_registration_count(epoch_b), 1);
}

/// Releasing one replica's epoch must never pull weights out from under a
/// sibling: the buffer survives until the LAST epoch referencing it goes.
#[test]
fn release_frees_a_shared_buffer_only_with_its_last_epoch() {
    let Some(b) = make_backend() else { return };
    let epoch_a = crate::gpu_backend::next_gpu_model_epoch();
    let epoch_b = crate::gpu_backend::next_gpu_model_epoch();
    let bytes = block_a();
    let (handle, _) = in_epoch(epoch_a, || b.upload_weights(&bytes));
    let handle = handle.expect("upload A");
    let (_, _) = in_epoch(epoch_b, || b.upload_weights(&bytes));

    assert_eq!(b.release_model_weights(epoch_a).expect("release A"), 0);
    assert_eq!(b.cached_weight_count(), 1, "epoch B still references it");
    assert_eq!(b.epoch_registration_count(epoch_a), 0);
    if b.is_accelerated() {
        let input = vec![1.0f32; 128];
        assert!(
            b.gemv_q1_g128_cached(handle, &input, 1, 128).is_ok(),
            "the surviving replica's handle must still resolve"
        );
    }

    assert_eq!(b.release_model_weights(epoch_b).expect("release B"), 1);
    assert_eq!(b.cached_weight_count(), 0);
    assert_eq!(b.cached_weight_bytes(), 0);
    assert!(
        b.gemv_q1_g128_cached(handle, &[1.0f32; 128], 1, 128)
            .is_err(),
        "a freed handle must not resolve to anything"
    );
    // Monotonic: freeing does not rewrite history.
    assert_eq!(b.uploaded_weight_bytes_total(), bytes.len() as u64);
}

#[test]
fn distinct_bytes_never_share_a_handle() {
    let Some(b) = make_backend() else { return };
    let epoch = crate::gpu_backend::next_gpu_model_epoch();
    let (handles, stats) = in_epoch(epoch, || {
        (
            b.upload_weights(&block_a()).expect("A"),
            b.upload_weights(&block_b()).expect("B"),
        )
    });
    assert_ne!(handles.0, handles.1);
    assert_eq!(b.cached_weight_count(), 2);
    assert_eq!(stats.fresh_buffers, 2);
    assert_eq!(stats.shared_buffers, 0);
}

/// The same input bytes through the raw and the ternary (SoA-reformatting)
/// upload are different device buffers read by different kernels; sharing
/// them would feed a Q1 kernel ternary SoA data or vice versa.
#[test]
fn raw_and_ternary_layouts_never_alias() {
    use half::f16;
    use oxibonsai_core::BlockTQ2_0_g128;
    let Some(b) = make_backend() else { return };
    let block = BlockTQ2_0_g128 {
        qs: [0x55u8; 32],
        d: f16::from_f32(1.0),
    };
    // The AoS bytes of the block, uploaded raw.
    let mut aos = Vec::with_capacity(34);
    aos.extend_from_slice(&block.qs);
    aos.extend_from_slice(&block.d.to_bits().to_le_bytes());
    let epoch = crate::gpu_backend::next_gpu_model_epoch();
    let (handles, _) = in_epoch(epoch, || {
        (
            b.upload_weights(&aos).expect("raw"),
            b.upload_weights_ternary(&[block]).expect("ternary"),
        )
    });
    assert_ne!(handles.0, handles.1);
    assert_eq!(b.cached_weight_count(), 2);
}

/// An upload made outside any scope has no known owner, so no release —
/// not even of the unattributed epoch itself — may free it.
#[test]
fn unattributed_uploads_survive_every_release() {
    let Some(b) = make_backend() else { return };
    let handle = b.upload_weights(&block_a()).expect("unscoped upload");
    assert_eq!(
        b.release_model_weights(crate::gpu_backend::UNATTRIBUTED_MODEL_EPOCH)
            .expect("release unattributed"),
        0
    );
    let unrelated = crate::gpu_backend::next_gpu_model_epoch();
    assert_eq!(
        b.release_model_weights(unrelated).expect("unknown epoch"),
        0
    );
    assert_eq!(b.cached_weight_count(), 1);

    // A scoped upload of the same bytes shares the unattributed buffer; its
    // release drops only its own reference.
    let epoch = crate::gpu_backend::next_gpu_model_epoch();
    let (shared, stats) = in_epoch(epoch, || b.upload_weights(&block_a()));
    assert_eq!(shared.expect("scoped upload"), handle);
    assert_eq!(stats.shared_buffers, 1);
    assert_eq!(b.release_model_weights(epoch).expect("release"), 0);
    assert_eq!(b.cached_weight_count(), 1);
}

/// Two uploads of the same bytes inside ONE epoch are two references; one
/// release drops both.
#[test]
fn one_epoch_uploading_twice_holds_two_references() {
    let Some(b) = make_backend() else { return };
    let epoch = crate::gpu_backend::next_gpu_model_epoch();
    let (handles, stats) = in_epoch(epoch, || {
        (
            b.upload_weights(&block_b()).expect("first"),
            b.upload_weights(&block_b()).expect("second"),
        )
    });
    assert_eq!(handles.0, handles.1);
    assert_eq!(stats.fresh_buffers, 1);
    assert_eq!(stats.shared_buffers, 1);
    assert_eq!(b.epoch_registration_count(epoch), 2);
    assert_eq!(b.release_model_weights(epoch).expect("release"), 1);
    assert_eq!(b.cached_weight_count(), 0);
}

#[test]
fn clear_weight_cache_also_forgets_every_registration() {
    let Some(b) = make_backend() else { return };
    let epoch = crate::gpu_backend::next_gpu_model_epoch();
    let (_, _) = in_epoch(epoch, || b.upload_weights(&block_a()));
    assert_eq!(b.clear_weight_cache().expect("clear"), 1);
    assert_eq!(b.epoch_registration_count(epoch), 0);
    assert_eq!(
        b.release_model_weights(epoch).expect("release after clear"),
        0
    );
}

/// With retention disabled nothing is shared, registered or counted.
#[test]
fn disabled_cache_never_registers_or_shares() {
    let Some(b) = make_backend() else { return };
    b.set_weight_cache_enabled(false);
    let epoch = crate::gpu_backend::next_gpu_model_epoch();
    let (handles, stats) = in_epoch(epoch, || {
        (
            b.upload_weights(&block_a()).expect("first"),
            b.upload_weights(&block_a()).expect("second"),
        )
    });
    assert_ne!(handles.0, handles.1, "no dedupe without retention");
    assert!(stats.is_empty());
    assert_eq!(b.epoch_registration_count(epoch), 0);
    assert_eq!(b.cached_weight_count(), 0);
    assert_eq!(b.uploaded_weight_bytes_total(), 0);
}

/// The trait surface `KernelDispatcher::gpu_backend()` exposes must reach the
/// same accounting (the engine's `Drop` only ever sees the trait object).
#[test]
fn trait_surface_reports_and_releases_the_same_state() {
    let Some(b) = make_backend() else { return };
    let epoch = crate::gpu_backend::next_gpu_model_epoch();
    let (_, _) = in_epoch(epoch, || b.upload_weights(&block_a()));
    let dyn_backend: &dyn GpuBackendTrait = &b;
    assert_eq!(dyn_backend.resident_weight_count(), 1);
    assert_eq!(dyn_backend.resident_weight_bytes(), block_a().len() as u64);
    assert_eq!(
        dyn_backend.weight_bytes_uploaded_total(),
        block_a().len() as u64
    );
    assert_eq!(dyn_backend.model_registration_count(epoch), 1);
    assert_eq!(dyn_backend.release_model(epoch).expect("release"), 1);
    assert_eq!(dyn_backend.resident_weight_count(), 0);
}
