//! `MET-08` acceptance: concurrent Metal sessions must produce **byte-identical**
//! results to the same work run sequentially.
//!
//! Before `MET-08` every GPU dispatch in the process went through one
//! `MetalGraph`: one command queue, one device KV cache, one set of full-layer
//! and prefill scratch buffers. The `MutexGuard`s taken at the top of a fused
//! forward were held through every layer's encode plus `commit()` and
//! `wait_until_completed()`, so two inference replicas could not overlap — and
//! because `GpuKvCache::matches` compares only the *shape*
//! `(n_layers, n_kv, max_seq, head_dim)`, two replicas of the same model would
//! have silently trampled each other's attention state. `engine_pool` clamped
//! GPU pools to a single replica for exactly that reason.
//!
//! The type is now split: a process-global `MetalDevice` (device, compiled
//! pipelines, weight cache — so N sessions still hold **1x** the weights) and a
//! per-session `MetalGraph` (its own command queue and its own workspace). This
//! file is the evidence that the split is real and safe:
//!
//! 1. two sessions are distinct objects that nonetheless share the weight cache;
//! 2. binding is scoped, re-entrant and thread-local;
//! 3. **50 iterations** of GPU GEMM run concurrently in two sessions give
//!    bit-for-bit the same floats as the sequential baseline;
//! 4. an unbound thread still gets the process-default session, so
//!    single-session behaviour is unchanged;
//! 5. the engine pool hands every replica its own session and concurrent
//!    generation matches isolated baselines token for token.
//!
//! Every GPU test no-ops on a host without a Metal device, so the suite is
//! green on CI machines with no GPU. All test names start with
//! `metal_concurrency_` so `cargo test -p oxibonsai-runtime --features metal
//! metal_concurrency` selects exactly this file.

#![cfg(all(feature = "metal", target_os = "macos"))]

use std::sync::Arc;

use oxibonsai_kernels::{MetalGraph, MetalWeightHandle};

/// Number of stress iterations per session in the byte-identity test.
///
/// The spec asks for 50: enough that a session-crossing scratch buffer or a
/// shared command queue would show up as a divergent float, fast enough to run
/// on every gate.
const STRESS_ITERATIONS: usize = 50;

/// GEMM shape for the stress test: `m x k` input against a `n_rows x k` weight.
const M: usize = 8;
/// Rows of the weight matrix (output width).
const N_ROWS: usize = 64;
/// Reduction dimension.
const K: usize = 128;

/// `true` when this host has no Metal device, so the GPU tests should no-op.
fn no_gpu() -> bool {
    MetalGraph::new_session().is_err()
}

/// A deterministic f32 weight matrix, distinct per `seed`.
fn weight_bytes(seed: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(N_ROWS * K * 4);
    for row in 0..N_ROWS {
        for col in 0..K {
            let v = ((row * 7 + col * 13 + seed * 31) % 19) as f32 * 0.125 - 1.0;
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    bytes
}

/// A deterministic `m x k` input, distinct per `seed`.
fn input(seed: usize) -> Vec<f32> {
    (0..M * K)
        .map(|i| ((i + seed * 17) % 23) as f32 * 0.0625 - 0.5)
        .collect()
}

/// Run one GEMM in `session` and return the output row-major.
fn gemm(session: &MetalGraph, weight: &MetalWeightHandle, seed: usize) -> Vec<f32> {
    let mut out = vec![0f32; M * N_ROWS];
    session
        .encode_gemm_f32(weight, &input(seed), &mut out, M, N_ROWS, K)
        .expect("encode_gemm_f32");
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Two sessions are distinct, and share the device + weight cache
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn metal_concurrency_sessions_are_distinct_but_share_the_weight_cache() {
    if no_gpu() {
        return;
    }
    let a = MetalGraph::new_session().expect("session a");
    let b = MetalGraph::new_session().expect("session b");
    assert_ne!(
        a.session_id(),
        b.session_id(),
        "each session must be individually identifiable"
    );

    // Sharing the weight cache is the entire point of splitting the device out:
    // N replicas of one model must hold 1x its weights, not Nx. An upload made
    // through session `a` is therefore visible in session `b`'s accounting, and
    // asking `b` for the same key hands back the *same* GPU buffer rather than
    // uploading a second copy.
    let key = 0x4d45_5430_3800_0001; // "MET08" + a slot nobody else uses
    let bytes = weight_bytes(1);
    let before = b.cached_weight_count().expect("cached count");
    let from_a = a.get_or_upload_weight(key, &bytes).expect("upload via a");
    let after_upload = b.cached_weight_count().expect("cached count");
    assert_eq!(
        after_upload,
        before + 1,
        "an upload in one session must land in the shared cache"
    );

    let uploads_before = b.weight_upload_count();
    let from_b = b.get_or_upload_weight(key, &bytes).expect("lookup via b");
    assert_eq!(
        b.weight_upload_count(),
        uploads_before,
        "a sibling session must hit the cache, not upload a second copy"
    );
    assert!(
        Arc::ptr_eq(&from_a, &from_b),
        "both sessions must be handed the same GPU buffer"
    );

    a.evict_f32_weight(key).expect("evict");
    assert_eq!(b.cached_weight_count().expect("cached count"), before);
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Binding is scoped, re-entrant, and thread-local
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn metal_concurrency_binding_is_scoped_and_reentrant() {
    if no_gpu() {
        return;
    }
    let outer = MetalGraph::new_session().expect("outer session");
    let inner = MetalGraph::new_session().expect("inner session");

    assert_eq!(
        MetalGraph::current_session_id(),
        None,
        "a fresh thread starts unbound"
    );

    MetalGraph::with_session(&outer, || {
        assert_eq!(MetalGraph::current_session_id(), Some(outer.session_id()));
        assert_eq!(
            MetalGraph::global().expect("global").session_id(),
            outer.session_id(),
            "global() must resolve to the bound session"
        );

        MetalGraph::with_session(&inner, || {
            assert_eq!(MetalGraph::current_session_id(), Some(inner.session_id()));
        });

        assert_eq!(
            MetalGraph::current_session_id(),
            Some(outer.session_id()),
            "leaving an inner scope must restore the outer binding"
        );

        // Re-binding the session that is already bound must not clear it when
        // that redundant scope ends.
        MetalGraph::with_session(&outer, || {
            assert_eq!(MetalGraph::current_session_id(), Some(outer.session_id()));
        });
        assert_eq!(MetalGraph::current_session_id(), Some(outer.session_id()));
    });

    assert_eq!(
        MetalGraph::current_session_id(),
        None,
        "the binding must not outlive its scope"
    );

    // Bindings are per thread: a sibling thread sees none of this one's.
    let seen = std::thread::spawn(MetalGraph::current_session_id)
        .join()
        .expect("join");
    assert_eq!(seen, None, "a binding must not leak across threads");
}

#[test]
fn metal_concurrency_unbound_threads_use_the_process_default_session() {
    if no_gpu() {
        return;
    }
    // The single-session invariant: with nobody binding anything, every caller
    // — on any thread — resolves to one session, exactly as before `MET-08`.
    // This is what makes a single-request run byte-identical to the old code.
    let here = MetalGraph::global().expect("global here").session_id();
    let there = std::thread::spawn(|| MetalGraph::global().expect("global there").session_id())
        .join()
        .expect("join");
    assert_eq!(
        here, there,
        "unbound threads must share the process-default session"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. The stress test: concurrent == sequential, bit for bit
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn metal_concurrency_two_sessions_match_two_sequential_runs_bitwise() {
    if no_gpu() {
        return;
    }

    // ── Baseline: both workloads run one after the other, in one session ──
    let baseline_session = MetalGraph::new_session().expect("baseline session");
    let mut baselines = Vec::new();
    for worker in 0..2usize {
        let handle = baseline_session
            .upload_weight(&weight_bytes(worker))
            .expect("upload baseline weight");
        let mut per_iteration = Vec::with_capacity(STRESS_ITERATIONS);
        for iteration in 0..STRESS_ITERATIONS {
            per_iteration.push(gemm(&baseline_session, &handle, worker * 1000 + iteration));
        }
        baselines.push(per_iteration);
    }

    // ── Concurrent: one session per worker, running at the same time ──
    let baselines = Arc::new(baselines);
    let mut threads = Vec::new();
    for worker in 0..2usize {
        let baselines = Arc::clone(&baselines);
        threads.push(std::thread::spawn(move || {
            let session = MetalGraph::new_session().expect("worker session");
            MetalGraph::with_session(&session, || {
                // Every `MetalGraph::global()` inside this closure — including
                // the ones the model and kernel stack make on its behalf — is
                // this worker's session.
                let bound = MetalGraph::global().expect("bound global");
                assert_eq!(bound.session_id(), session.session_id());

                let handle = session
                    .upload_weight(&weight_bytes(worker))
                    .expect("upload worker weight");
                for iteration in 0..STRESS_ITERATIONS {
                    let got = gemm(&session, &handle, worker * 1000 + iteration);
                    assert_eq!(
                        got, baselines[worker][iteration],
                        "worker {worker} iteration {iteration} diverged from the sequential \
                         baseline — cross-session scratch or queue interference"
                    );
                }
            });
            session.session_id()
        }));
    }

    let ids: Vec<u64> = threads
        .into_iter()
        .map(|t| t.join().expect("worker thread"))
        .collect();
    assert_ne!(ids[0], ids[1], "the two workers must not share a session");
}

#[test]
fn metal_concurrency_many_sessions_stay_independent_under_load() {
    if no_gpu() {
        return;
    }
    // Four workers, deliberately above the default session ceiling used for
    // pool sizing: creating sessions is never refused (the ceiling is a
    // *pool-sizing* policy, not a hard limit), and each one still computes its
    // own workload correctly while the others hammer the same device.
    let workers = 4usize;
    let reference = MetalGraph::new_session().expect("reference session");
    let expected: Vec<Vec<f32>> = (0..workers)
        .map(|w| {
            let handle = reference
                .upload_weight(&weight_bytes(w + 7))
                .expect("upload reference weight");
            gemm(&reference, &handle, w + 7)
        })
        .collect();

    let expected = Arc::new(expected);
    let mut threads = Vec::new();
    for worker in 0..workers {
        let expected = Arc::clone(&expected);
        threads.push(std::thread::spawn(move || {
            let session = MetalGraph::new_session().expect("worker session");
            let handle = session
                .upload_weight(&weight_bytes(worker + 7))
                .expect("upload worker weight");
            for _ in 0..STRESS_ITERATIONS / 5 {
                assert_eq!(
                    gemm(&session, &handle, worker + 7),
                    expected[worker],
                    "worker {worker} diverged under concurrent load"
                );
            }
        }));
    }
    for t in threads {
        t.join().expect("worker thread");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Pool sizing follows the session ceiling
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn metal_concurrency_gpu_pool_sizing_follows_the_session_ceiling() {
    use oxibonsai_kernels::KernelTier;
    use oxibonsai_runtime::engine_pool::{gpu_max_replicas, resolve_pool_sizing_with_gpu_max};

    // The GPU tier is no longer clamped to one replica: it is capped by how
    // many sessions the host can afford (one device KV cache each).
    let sizing = resolve_pool_sizing_with_gpu_max(Some(3), KernelTier::Gpu, 3);
    assert_eq!(sizing.effective, 3);
    assert!(!sizing.clamped_by_gpu_tier);

    // Above the ceiling the request is capped *and reported* as capped, so an
    // admission controller can shed rather than queue invisibly (`perf-M1`).
    let capped = resolve_pool_sizing_with_gpu_max(Some(9), KernelTier::Gpu, 2);
    assert_eq!(capped.effective, 2);
    assert!(capped.clamped_by_gpu_tier);
    assert_eq!(capped.gpu_max, Some(2));

    assert!(
        gpu_max_replicas() >= 1,
        "a pool must have at least a replica"
    );
    assert_eq!(
        gpu_max_replicas(),
        MetalGraph::max_sessions().max(1),
        "pool sizing must follow the kernels crate's session ceiling"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. The engine pool: a session per replica, and identical output under load
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn metal_concurrency_pool_replicas_hold_their_own_sessions_and_agree_with_baselines() {
    use oxibonsai_core::config::Qwen3Config;
    use oxibonsai_runtime::engine::InferenceEngine;
    use oxibonsai_runtime::engine_pool::EnginePool;
    use oxibonsai_runtime::sampling::SamplingParams;

    // This mirrors the server's real shape: acquire a lease on the async
    // runtime, then move it into `spawn_blocking` and generate there — which
    // is precisely why the lease binds its session in `Deref` (on the thread
    // that *uses* it) rather than in `acquire` (on the thread that took it).
    //
    // Two things are asserted: (a) inside the blocking thread the bound
    // session is the lease's own, and it is released again when the lease
    // drops; (b) three replicas generating simultaneously each reproduce the
    // token sequence they produce alone. On a Metal host `auto_detect` still
    // resolves this synthetic config to a CPU tier, so (a) is vacuous there
    // (`None == None`) and covered non-vacuously by
    // `engine_pool`'s own `gpu_session_binds_and_releases_on_the_using_thread`;
    // (b) holds on every tier.
    let config = Qwen3Config::tiny_test();
    let params = SamplingParams::default();
    let seed = 42u64;
    let prompts: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
    let max_tokens = 4usize;

    // Isolated baselines: one fresh engine per prompt, run alone.
    let baselines: Vec<Vec<u32>> = prompts
        .iter()
        .map(|p| {
            let mut engine = InferenceEngine::new(config.clone(), params.clone(), seed);
            engine.generate(p, max_tokens).expect("baseline generate")
        })
        .collect();

    let engines: Vec<InferenceEngine<'static>> = (0..prompts.len())
        .map(|_| InferenceEngine::new(config.clone(), params.clone(), seed))
        .collect();
    let pool = EnginePool::new(engines);

    // Lease every replica *before* any generation starts, so each task holds a
    // distinct, untouched replica for its whole life — otherwise a fast task
    // could return its replica and a later one reuse it, comparing a second
    // run against a first-run baseline.
    let mut leases = Vec::new();
    for _ in 0..prompts.len() {
        leases.push(pool.acquire().await.expect("acquire"));
    }
    let ids: Vec<Option<u64>> = leases.iter().map(|l| l.gpu_session_id()).collect();
    for (i, a) in ids.iter().enumerate() {
        for b in ids.iter().skip(i + 1) {
            if let (Some(a), Some(b)) = (a, b) {
                assert_ne!(a, b, "two live replicas share one Metal session");
            }
        }
    }

    let mut handles = Vec::new();
    for (lease, prompt) in leases.into_iter().zip(prompts.clone()) {
        handles.push(tokio::task::spawn_blocking(move || {
            let mut lease = lease;
            let expected_session = lease.gpu_session_id();
            let out = lease.generate(&prompt, max_tokens).expect("generate");
            // The lease bound its session on the blocking thread, not on the
            // async worker that acquired it.
            let bound_during = MetalGraph::current_session_id();
            drop(lease);
            // ...and released it again on the way out, so an unleased dispatch
            // on this thread falls back to the process-default session.
            let bound_after = MetalGraph::current_session_id();
            (out, expected_session, bound_during, bound_after)
        }));
    }

    for (i, h) in handles.into_iter().enumerate() {
        let (got, expected_session, bound_during, bound_after) = h.await.expect("join");
        assert_eq!(
            bound_during, expected_session,
            "request {i} did not run in its replica's session"
        );
        assert_eq!(
            bound_after, None,
            "request {i} left its session bound after the lease dropped"
        );
        assert_eq!(
            got, baselines[i],
            "concurrent request {i} diverged from its isolated baseline"
        );
    }
}
