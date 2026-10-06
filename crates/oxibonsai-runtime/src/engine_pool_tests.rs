//! Unit tests for `engine_pool.rs` (sibling file, declared there via
//! `#[path]`, so `super` still names that module and every test keeps its
//! `engine_pool::tests::` path).

use super::*;
use crate::sampling::SamplingParams;
use oxibonsai_core::config::Qwen3Config;
use std::time::Duration;

fn tiny_engine() -> InferenceEngine<'static> {
    InferenceEngine::new(Qwen3Config::tiny_test(), SamplingParams::default(), 42)
}

// ── synthetic GGUF fixture (for the shared-embd pool test) ───────────────
//
// A minimal 2-layer, fully-quantized GGUF (h=128, inter=256, vocab=32) that
// `BonsaiModel::from_gguf` can load on any CPU tier. Attention/FFN are
// Q1_0_g128 and the LM head is Q1_0_g128; the token embedding is F32. This
// is the same shape family used by the model crate's ternary integration
// fixture, reproduced compactly here so the runtime pool builder can be
// exercised end-to-end (it needs a real on-disk GGUF path).

fn q1_0_g128_data(num_weights: usize) -> Vec<u8> {
    let num_blocks = num_weights / 128;
    let scale = half::f16::ONE.to_le_bytes();
    let mut data = Vec::with_capacity(num_blocks * 18);
    for _ in 0..num_blocks {
        data.extend_from_slice(&scale);
        data.extend_from_slice(&[0xFFu8; 16]);
    }
    data
}

fn build_tiny_gguf_bytes() -> Vec<u8> {
    build_tiny_gguf_bytes_with(Vec::new())
}

/// [`build_tiny_gguf_bytes`] plus caller-supplied metadata (e.g. a
/// `qwen3.rope.scaling.*` declaration for the `--rope-scaling` tests).
pub(crate) fn build_tiny_gguf_bytes_with(
    extra_metadata: Vec<(
        &'static str,
        oxibonsai_core::gguf::writer::MetadataWriteValue,
    )>,
) -> Vec<u8> {
    use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue, TensorEntry, TensorType};

    let h: usize = 128;
    let inter: usize = 256;
    let num_layers: usize = 2;
    let nq: usize = 4;
    let nkv: usize = 2;
    let hd: usize = 32;
    let vocab: usize = 32;

    let mut w = GgufWriter::new();
    w.add_metadata(
        "general.architecture",
        MetadataWriteValue::Str("qwen3".into()),
    );
    w.add_metadata("general.name", MetadataWriteValue::Str("TinyPool".into()));
    w.add_metadata("qwen3.embedding_length", MetadataWriteValue::U32(h as u32));
    w.add_metadata(
        "qwen3.block_count",
        MetadataWriteValue::U32(num_layers as u32),
    );
    w.add_metadata(
        "qwen3.attention.head_count",
        MetadataWriteValue::U32(nq as u32),
    );
    w.add_metadata(
        "qwen3.attention.head_count_kv",
        MetadataWriteValue::U32(nkv as u32),
    );
    w.add_metadata(
        "qwen3.feed_forward_length",
        MetadataWriteValue::U32(inter as u32),
    );
    w.add_metadata("qwen3.vocab_size", MetadataWriteValue::U32(vocab as u32));
    w.add_metadata("qwen3.context_length", MetadataWriteValue::U32(512));
    w.add_metadata(
        "qwen3.attention.layer_norm_rms_epsilon",
        MetadataWriteValue::F32(1e-6),
    );
    w.add_metadata("qwen3.rope.freq_base", MetadataWriteValue::F32(10_000.0));
    for (key, value) in extra_metadata {
        w.add_metadata(key, value);
    }

    let f32_ones = |n: usize| -> Vec<u8> {
        let mut v = Vec::with_capacity(n * 4);
        for _ in 0..n {
            v.extend_from_slice(&1.0_f32.to_le_bytes());
        }
        v
    };

    w.add_tensor(TensorEntry {
        name: "token_embd.weight".into(),
        shape: vec![h as u64, vocab as u64],
        tensor_type: TensorType::F32,
        data: f32_ones(vocab * h),
    });
    w.add_tensor(TensorEntry {
        name: "output_norm.weight".into(),
        shape: vec![h as u64],
        tensor_type: TensorType::F32,
        data: f32_ones(h),
    });
    w.add_tensor(TensorEntry {
        name: "output.weight".into(),
        shape: vec![h as u64, vocab as u64],
        tensor_type: TensorType::Q1_0G128,
        data: q1_0_g128_data(vocab * h),
    });

    for layer in 0..num_layers {
        let pfx = format!("blk.{layer}");
        for suffix in ["attn_norm.weight", "ffn_norm.weight"] {
            w.add_tensor(TensorEntry {
                name: format!("{pfx}.{suffix}"),
                shape: vec![h as u64],
                tensor_type: TensorType::F32,
                data: f32_ones(h),
            });
        }
        for suffix in ["attn_q_norm.weight", "attn_k_norm.weight"] {
            w.add_tensor(TensorEntry {
                name: format!("{pfx}.{suffix}"),
                shape: vec![hd as u64],
                tensor_type: TensorType::F32,
                data: f32_ones(hd),
            });
        }
        let q1 = |name: &str, shape: Vec<u64>, n: usize| TensorEntry {
            name: name.to_string(),
            shape,
            tensor_type: TensorType::Q1_0G128,
            data: q1_0_g128_data(n),
        };
        w.add_tensor(q1(
            &format!("{pfx}.attn_q.weight"),
            vec![h as u64, (nq * hd) as u64],
            nq * hd * h,
        ));
        w.add_tensor(q1(
            &format!("{pfx}.attn_k.weight"),
            vec![h as u64, (nkv * hd) as u64],
            nkv * hd * h,
        ));
        w.add_tensor(q1(
            &format!("{pfx}.attn_v.weight"),
            vec![h as u64, (nkv * hd) as u64],
            nkv * hd * h,
        ));
        w.add_tensor(q1(
            &format!("{pfx}.attn_output.weight"),
            vec![(nq * hd) as u64, h as u64],
            h * nq * hd,
        ));
        w.add_tensor(q1(
            &format!("{pfx}.ffn_gate.weight"),
            vec![h as u64, inter as u64],
            inter * h,
        ));
        w.add_tensor(q1(
            &format!("{pfx}.ffn_up.weight"),
            vec![h as u64, inter as u64],
            inter * h,
        ));
        w.add_tensor(q1(
            &format!("{pfx}.ffn_down.weight"),
            vec![inter as u64, h as u64],
            h * inter,
        ));
    }

    w.to_bytes().expect("GgufWriter::to_bytes")
}

// ── resolve_pool_size ────────────────────────────────────────────────

#[test]
fn resolve_pool_size_explicit_cpu() {
    // On a CPU tier, an explicit request is honored (clamped to >= 1).
    let tier = oxibonsai_kernels::KernelTier::Reference;
    assert_eq!(resolve_pool_size(Some(8), tier), 8);
    assert_eq!(resolve_pool_size(Some(1), tier), 1);
    // Zero is clamped up to the floor of 1.
    assert_eq!(resolve_pool_size(Some(0), tier), 1);
}

#[test]
fn resolve_pool_size_default_cpu() {
    let tier = oxibonsai_kernels::KernelTier::Reference;
    assert_eq!(resolve_pool_size(None, tier), default_cpu_pool_size());
}

#[test]
fn default_cpu_pool_size_in_range() {
    let n = default_cpu_pool_size();
    assert!((1..=4).contains(&n), "expected 1..=4, got {n}");
}

#[cfg(any(feature = "metal", feature = "native-cuda"))]
#[test]
fn resolve_pool_size_gpu_is_capped_by_the_session_ceiling() {
    // `MET-08` changed this rule deliberately: the GPU tier used to be a
    // hard clamp to 1 because decode funnelled through a process-global
    // graph with one KV cache. A Metal replica now owns its own session,
    // so the tier admits up to `gpu_max` replicas — the bound is device
    // memory (one KV cache per session), not correctness.
    let tier = oxibonsai_kernels::KernelTier::Gpu;
    assert_eq!(
        resolve_pool_sizing_with_gpu_max(Some(8), tier, 3).effective,
        3
    );
    assert_eq!(
        resolve_pool_sizing_with_gpu_max(Some(2), tier, 3).effective,
        2
    );
    assert_eq!(
        resolve_pool_sizing_with_gpu_max(Some(1), tier, 3).effective,
        1
    );
    // An unspecified request stays conservative: one replica.
    assert_eq!(resolve_pool_sizing_with_gpu_max(None, tier, 3).effective, 1);
    // A ceiling of 1 (a CUDA-only build) reproduces the old hard clamp.
    assert_eq!(
        resolve_pool_sizing_with_gpu_max(Some(8), tier, 1).effective,
        1
    );
    // A zero/absurd ceiling can never yield an empty pool.
    assert_eq!(
        resolve_pool_sizing_with_gpu_max(Some(8), tier, 0).effective,
        1
    );
    // The live wiring agrees with the pure function.
    assert_eq!(
        resolve_pool_size(Some(8), tier),
        resolve_pool_sizing_with_gpu_max(Some(8), tier, gpu_max_replicas()).effective
    );
}

#[cfg(any(feature = "metal", feature = "native-cuda"))]
#[test]
fn gpu_max_replicas_is_at_least_one() {
    // A pool of zero replicas would deadlock every request on `acquire`.
    assert!(gpu_max_replicas() >= 1);
}

// ── lease / pool mechanics ───────────────────────────────────────────

#[tokio::test]
async fn pool_size_reflects_input() {
    let pool = EnginePool::new(vec![tiny_engine(), tiny_engine()]);
    assert_eq!(pool.size(), 2);
}

#[tokio::test]
async fn acquire_blocks_when_exhausted_then_resumes_on_drop() {
    let pool = EnginePool::new(vec![tiny_engine(), tiny_engine()]);

    // Take both engines.
    let lease_a = pool.acquire().await.expect("acquire a");
    let lease_b = pool.acquire().await.expect("acquire b");

    // Idle is now empty and no permits remain.
    assert_eq!(pool.sem.available_permits(), 0);
    {
        let idle = pool.idle.lock().expect("lock idle");
        assert!(idle.is_empty(), "idle should be empty with 2/2 checked out");
    }

    // A third acquire must NOT resolve while both leases are held.
    let pending = pool.acquire();
    let timed_out = tokio::time::timeout(Duration::from_millis(150), pending).await;
    assert!(
        timed_out.is_err(),
        "third acquire resolved while pool was exhausted"
    );

    // Returning one engine must let a waiting acquire proceed.
    drop(lease_a);
    let lease_c = tokio::time::timeout(Duration::from_millis(500), pool.acquire())
        .await
        .expect("acquire should resolve after a lease is dropped")
        .expect("acquire c");

    // Drop the rest; the pool returns to full availability.
    drop(lease_b);
    drop(lease_c);
    assert_eq!(pool.sem.available_permits(), 2);
    {
        let idle = pool.idle.lock().expect("lock idle");
        assert_eq!(idle.len(), 2, "all engines should be back in the pool");
    }
}

// ── single-request golden: byte-identical behavior ───────────────────

#[tokio::test]
async fn single_element_pool_is_byte_identical_to_direct_engine() {
    // The hard invariant: a 1-element pool that acquires and calls
    // `generate_with_params` must produce the EXACT same token vector as a
    // fresh engine with the same config/seed/params calling the same
    // method directly.
    let config = Qwen3Config::tiny_test();
    let params = SamplingParams::default();
    let seed = 42u64;
    let prompt: Vec<u32> = vec![151644, 872, 9707, 11];
    let max_tokens = 8usize;

    // Direct engine baseline.
    let mut direct = InferenceEngine::new(config.clone(), params.clone(), seed);
    let direct_out = direct
        .generate_with_params(&prompt, max_tokens, &params)
        .expect("direct generate");

    // 1-element pool.
    let pool = EnginePool::new(vec![InferenceEngine::new(
        config.clone(),
        params.clone(),
        seed,
    )]);
    let mut lease = pool.acquire().await.expect("acquire");
    let pool_out = lease
        .generate_with_params(&prompt, max_tokens, &params)
        .expect("pool generate");

    assert_eq!(
        direct_out, pool_out,
        "1-element pool output diverged from direct engine — byte-identity broken"
    );
}

// ── concurrent isolation: no KV / RNG cross-talk ─────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_leases_match_isolated_baselines() {
    // NOTE on test strength: `Qwen3Config::tiny_test()` builds a model with
    // zero-initialized weights and no transformer blocks, so its forward
    // pass is prompt-INDEPENDENT (every prompt yields the same logits ->
    // same greedy token). That makes "distinct prompts => distinct outputs"
    // impossible to assert here. What we CAN — and do — assert is the
    // structural isolation guarantee: each concurrently-leased engine
    // produces output bit-identical to the SAME prompt run alone on a fresh
    // single engine. If concurrent leases shared/corrupted KV or RNG state,
    // the concurrent outputs would diverge from their isolated baselines.
    //
    // A richer cross-talk test (distinct prompts => distinct outputs)
    // requires non-degenerate weights; that variant is best built behind
    // `#[cfg(all(feature = "metal", target_os = "macos"))]` using the
    // synthetic ternary GGUF fixture from
    // `crates/oxibonsai-model/tests/metal_prefill_ternary_parity_tests.rs`
    // with a couple of `KernelTier::Reference` replicas. See task notes.
    use std::sync::Arc as StdArc;

    let config = Qwen3Config::tiny_test();
    // GREEDY params (temperature 0) make outputs deterministic and remove
    // any dependence on RNG ordering, isolating the KV-cache question.
    let params = SamplingParams {
        temperature: 0.0,
        ..SamplingParams::default()
    };
    let seed = 42u64;
    let max_tokens = 6usize;

    let prompts: Vec<Vec<u32>> = vec![
        vec![151644, 872],
        vec![151644, 9707, 11, 1879],
        vec![151644, 1986, 374, 264, 1273],
        vec![151644, 264],
    ];

    // Isolated baselines: each prompt alone on a fresh single engine.
    let mut baselines = Vec::with_capacity(prompts.len());
    for p in &prompts {
        let mut e = InferenceEngine::new(config.clone(), params.clone(), seed);
        let out = e
            .generate_with_params(p, max_tokens, &params)
            .expect("baseline generate");
        baselines.push(out);
    }

    // Pool with one replica per prompt so all run truly concurrently.
    let engines: Vec<InferenceEngine<'static>> = (0..prompts.len())
        .map(|_| InferenceEngine::new(config.clone(), params.clone(), seed))
        .collect();
    let pool = EnginePool::new(engines);

    let params = StdArc::new(params);
    let mut handles = Vec::with_capacity(prompts.len());
    for p in prompts.clone() {
        let pool = StdArc::clone(&pool);
        let params = StdArc::clone(&params);
        handles.push(tokio::spawn(async move {
            let mut lease = pool.acquire().await.expect("acquire");
            lease
                .generate_with_params(&p, max_tokens, &params)
                .expect("concurrent generate")
        }));
    }

    for (i, h) in handles.into_iter().enumerate() {
        let got = h.await.expect("task join");
        assert_eq!(
            got, baselines[i],
            "concurrent task {i} diverged from its isolated baseline — KV/RNG cross-talk"
        );
    }
}

// ── shared token-embedding Arc across pool replicas ──────────────────────

#[tokio::test]
async fn pool_replicas_share_one_token_embd_allocation() {
    // The end-to-end Part-B proof: `build_pool_from_gguf` must build all
    // replicas sharing ONE `Arc<[f32]>` token-embedding table (collapsing N
    // duplicate ~1.16 GiB allocations into one for the real 1.7B), while
    // each replica keeps its own KV cache.
    //
    // The synthetic fixture loads on any CPU tier; `from_gguf` auto-detects
    // the kernel. On a GPU tier the pool clamps to size 1 (a process-global
    // singleton), in which case the multi-replica ptr-equality assertion is
    // vacuous — so we skip it and only sanity-check the single replica. On
    // this Mac `auto_detect` returns NEON (a CPU tier), so the multi-replica
    // path is the one normally exercised here.
    let bytes = build_tiny_gguf_bytes();
    let path = {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "oxibonsai_pool_shared_embd_{}.gguf",
            std::process::id()
        ));
        p
    };
    std::fs::write(&path, &bytes).expect("write temp GGUF");

    let (pool, _tier, size) =
        build_pool_from_gguf(&path, SamplingParams::default(), 42, 512, Some(3))
            .expect("build_pool_from_gguf");

    // Clean up the temp file now that the GGUF is mmapped + leaked into the
    // pool (the leaked mmap keeps the bytes alive regardless of the file).
    let _ = std::fs::remove_file(&path);

    if size <= 1 {
        // GPU tier (or single-core host): only one replica exists, so there
        // is nothing to share. Just confirm the lone replica is usable.
        let lease = pool.acquire().await.expect("acquire sole replica");
        let embd = lease.model_token_embd();
        assert!(!embd.is_empty(), "token_embd must be populated");
        return;
    }

    // Acquire ALL replicas at once so we can compare every replica's
    // `token_embd` handle simultaneously. With `size` permits this never
    // blocks.
    let mut leases = Vec::with_capacity(size);
    for _ in 0..size {
        leases.push(pool.acquire().await.expect("acquire replica"));
    }

    // Every replica's token_embd must be the SAME allocation.
    let first_embd = leases[0].model_token_embd();
    for (i, lease) in leases.iter().enumerate().skip(1) {
        let other = lease.model_token_embd();
        assert!(
            Arc::ptr_eq(&first_embd, &other),
            "replica #{i} token_embd is a different allocation — sharing broken"
        );
    }

    // KV caches must be DISTINCT per replica (per-request mutable state).
    let kv_ptrs: Vec<*const _> = leases
        .iter()
        .map(|l| {
            l.dense_model()
                .expect("the Q1 fixture is a dense model")
                .kv_cache() as *const _
        })
        .collect();
    for i in 0..kv_ptrs.len() {
        for j in (i + 1)..kv_ptrs.len() {
            assert_ne!(
                kv_ptrs[i], kv_ptrs[j],
                "replicas #{i} and #{j} share a KV cache — isolation broken"
            );
        }
    }

    // Strong count: all `size` replicas alias the one allocation. We hold
    // `first_embd` plus `size` replica-held clones; the per-replica handle
    // pulled inside the loop above has been dropped. So the count is
    // `size + 1`.
    assert_eq!(
        Arc::strong_count(&first_embd),
        size + 1,
        "expected {size} replicas + the local handle to alias one allocation"
    );
}

// ── perf-M1: the real capacity must be visible to admission control ──

#[test]
fn admission_limit_is_derived_from_the_effective_pool_size() {
    let pool = EnginePool::new(vec![tiny_engine(), tiny_engine()]);
    assert_eq!(pool.size(), 2);
    // No queueing: admit only what can start immediately.
    assert_eq!(pool.admission_limit(0), 2);
    // One warm request per replica.
    assert_eq!(pool.admission_limit(1), 4);
    // A hostile depth must saturate, not wrap to a small limit.
    assert!(pool.admission_limit(usize::MAX) >= 2);
}

#[test]
fn admission_limit_is_never_zero() {
    let pool = EnginePool::new(vec![tiny_engine()]);
    assert_eq!(pool.admission_limit(0), 1);
}

/// `SV-09` lifecycle: a request's cancellation token must not survive
/// the lease that armed it. Without this, a replica that served one
/// cancelled (or timed-out) request would instantly cancel the next
/// request it is handed — a one-request outage per timeout.
#[tokio::test]
async fn a_cancelled_request_does_not_poison_the_next_lease() {
    let pool = EnginePool::new(vec![tiny_engine()]);

    let token = {
        let mut lease = pool.acquire().await.expect("acquire");
        // The server's shape: reset (as `run_blocking_generation` does),
        // then arm, then generate.
        lease.reset();
        let token = lease.arm_cancellation();
        token.cancel();
        assert!(lease.is_cancelled());
        token
    };
    // The token itself stays cancelled for whoever still holds it ...
    assert!(token.is_cancelled());

    // ... but the replica handed to the next request is not armed at all.
    let next = pool.acquire().await.expect("re-acquire");
    assert!(
        next.cancellation_token().is_none(),
        "the returned replica must carry no token from the previous request"
    );
    assert!(!next.is_cancelled());
}

/// A cancel that arrives late — after its request's lease went back to the
/// pool and the next request armed the same replica (an abandoned
/// request's guard firing after the answer was already in hand) — reaches
/// only the finished request's token: the next request's generation runs to
/// its budget.
#[tokio::test]
async fn a_late_cancel_does_not_reach_the_next_request_on_the_replica() {
    let pool = EnginePool::new(vec![tiny_engine()]);
    let finished = {
        let mut lease = pool.acquire().await.expect("acquire");
        lease.reset();
        let token = lease.arm_cancellation();
        let tokens = lease.generate(&[1, 2, 3], 2).expect("generation");
        assert_eq!(tokens.len(), 2);
        token
    };

    let mut next = pool.acquire().await.expect("the same (only) replica");
    next.reset();
    let next_token = next.arm_cancellation();
    finished.cancel();
    assert!(!next_token.is_cancelled());
    assert!(!next.is_cancelled());
    let tokens = next.generate(&[1, 2, 3], 4).expect("generation");
    assert_eq!(tokens.len(), 4, "the next request is not cut short");
}

#[tokio::test]
async fn idle_count_tracks_outstanding_leases() {
    let pool = EnginePool::new(vec![tiny_engine(), tiny_engine()]);
    assert_eq!(pool.idle_count(), 2);
    let lease = pool.acquire().await.expect("acquire");
    assert_eq!(pool.idle_count(), 1);
    drop(lease);
    assert_eq!(pool.idle_count(), 2);
}

#[test]
fn pool_sizing_reports_the_host_default_when_nothing_was_requested() {
    let sizing = resolve_pool_sizing(None, oxibonsai_kernels::KernelTier::Reference);
    assert_eq!(sizing.effective, default_cpu_pool_size());
    assert_eq!(sizing.requested, None);
    assert!(!sizing.clamped_by_gpu_tier);
    assert!(
        sizing.reason().contains("host default"),
        "{}",
        sizing.reason()
    );
}

#[test]
fn pool_sizing_reports_an_honoured_request_on_a_cpu_tier() {
    let sizing = resolve_pool_sizing(Some(3), oxibonsai_kernels::KernelTier::Reference);
    assert_eq!(sizing.effective, 3);
    assert!(!sizing.clamped_by_gpu_tier);
    assert_eq!(
        sizing.effective,
        resolve_pool_size(Some(3), oxibonsai_kernels::KernelTier::Reference)
    );
}

#[cfg(any(feature = "metal", feature = "native-cuda"))]
#[test]
fn pool_sizing_records_the_gpu_clamp_instead_of_hiding_it() {
    // `perf-M1`: a request above the ceiling must still be *reported* as a
    // clamp, so an admission controller can shed instead of queueing 8
    // requests behind 2 replicas.
    let tier = oxibonsai_kernels::KernelTier::Gpu;
    let sizing = resolve_pool_sizing_with_gpu_max(Some(8), tier, 2);
    assert_eq!(sizing.effective, 2);
    assert_eq!(sizing.gpu_max, Some(2));
    assert!(
        sizing.clamped_by_gpu_tier,
        "a server must be able to tell a clamp from an honoured request"
    );
    let reason = sizing.reason();
    assert!(reason.contains("requested 8"), "{reason}");
    assert!(reason.contains("GPU tier"), "{reason}");
    assert!(reason.contains("clamped to 2"), "{reason}");

    // A request at or below the ceiling is honoured, not clamped.
    let honoured = resolve_pool_sizing_with_gpu_max(Some(2), tier, 2);
    assert_eq!(honoured.effective, 2);
    assert!(!honoured.clamped_by_gpu_tier);

    // An unspecified request that lands on 1 anyway is not a clamp.
    let default_sizing = resolve_pool_sizing_with_gpu_max(None, tier, 4);
    assert_eq!(default_sizing.effective, 1);
    assert!(!default_sizing.clamped_by_gpu_tier);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn gpu_session_binds_and_releases_on_the_using_thread() {
    // Direct, non-vacuous cover for the lease's `Deref`-binds /
    // `Drop`-releases contract (`MET-08`). The pool fixtures in this crate
    // all resolve to a CPU tier, where `GpuSession` is empty, so the
    // binding itself is exercised here against a real session instead.
    let Ok(session) = oxibonsai_kernels::MetalGraph::new_session() else {
        return; // no Metal device on this host
    };
    let id = session.session_id();
    let gpu = GpuSession(Some(session));

    assert_eq!(oxibonsai_kernels::MetalGraph::current_session_id(), None);
    gpu.bind();
    assert_eq!(
        oxibonsai_kernels::MetalGraph::current_session_id(),
        Some(id),
        "the replica's session must be bound to the thread using it"
    );
    // Re-binding the same session is idempotent (this runs on every deref).
    gpu.bind();
    assert_eq!(
        oxibonsai_kernels::MetalGraph::current_session_id(),
        Some(id)
    );
    assert_eq!(
        oxibonsai_kernels::MetalGraph::global()
            .expect("global")
            .session_id(),
        id,
        "global() must resolve to the leased replica's session"
    );

    gpu.release();
    assert_eq!(
        oxibonsai_kernels::MetalGraph::current_session_id(),
        None,
        "returning a replica must release its binding"
    );
    // Releasing a binding this thread no longer holds is a no-op.
    gpu.release();
    assert_eq!(oxibonsai_kernels::MetalGraph::current_session_id(), None);
}

#[tokio::test]
async fn every_replica_gets_its_own_gpu_session() {
    // `MET-08`: two replicas leased at once must never share a session —
    // a shared session means a shared device KV cache, which is exactly
    // the state two concurrent decodes would trample. Off the Metal tier
    // (this fixture is a CPU-tier `tiny_test` engine, and CI hosts have no
    // GPU at all) there is no session and both report `None`, which the
    // assertion below tolerates.
    let pool = EnginePool::new(vec![tiny_engine(), tiny_engine()]);
    let a = pool.acquire().await.expect("acquire a");
    let b = pool.acquire().await.expect("acquire b");
    match (a.gpu_session_id(), b.gpu_session_id()) {
        (Some(x), Some(y)) => assert_ne!(x, y, "two live replicas share one Metal session"),
        (None, None) => {}
        other => panic!("replicas disagree about having a GPU session: {other:?}"),
    }

    // A replica keeps its session across leases: the id it reports after
    // being returned and re-acquired is the one it had before.
    let a_id = a.gpu_session_id();
    drop(a);
    let a_again = pool.acquire().await.expect("re-acquire a");
    assert_eq!(a_again.gpu_session_id(), a_id);
}

// ── `--rope-scaling auto|on|off` ──────────────────────────────────────

/// The exact YaRN declaration `models/Bonsai-8B.gguf` carries, on the
/// tiny dense fixture.
pub(crate) fn yarn_metadata() -> Vec<(
    &'static str,
    oxibonsai_core::gguf::writer::MetadataWriteValue,
)> {
    use oxibonsai_core::gguf::writer::MetadataWriteValue;
    vec![
        (
            "qwen3.rope.scaling.type",
            MetadataWriteValue::Str("yarn".to_string()),
        ),
        ("qwen3.rope.scaling.factor", MetadataWriteValue::F32(4.0)),
        (
            "qwen3.rope.scaling.original_context_length",
            MetadataWriteValue::U32(128),
        ),
    ]
}

pub(crate) fn engine_rope_scaling(
    engine: &InferenceEngine<'_>,
) -> oxibonsai_core::config::RopeScaling {
    engine
        .dense_model()
        .expect("the tiny fixture is a dense model")
        .config()
        .rope_scaling
        .clone()
}

/// A tiny GGUF declaring YaRN — `auto` applies it, `off` builds the model
/// with plain RoPE, and the choice does not leak into a later `auto` load on
/// the same thread.
#[test]
fn rope_override_on_a_yarn_gguf_auto_applies_and_off_does_not() {
    use oxibonsai_core::config::{RopeScaling, RopeScalingOverride};
    let bytes = build_tiny_gguf_bytes_with(yarn_metadata());
    let gguf = oxibonsai_core::gguf::reader::GgufFile::parse(&bytes).expect("parse fixture");

    let auto = InferenceEngine::from_gguf_with_backend_and_rope(
        &gguf,
        SamplingParams::default(),
        42,
        64,
        crate::engine_seam::Backend::Cpu,
        RopeScalingOverride::Auto,
    )
    .expect("auto load");
    assert!(
        matches!(engine_rope_scaling(&auto), RopeScaling::Yarn { factor, .. } if factor == 4.0),
        "auto must honour the declared YaRN"
    );

    let off = InferenceEngine::from_gguf_with_backend_and_rope(
        &gguf,
        SamplingParams::default(),
        42,
        64,
        crate::engine_seam::Backend::Cpu,
        RopeScalingOverride::Off,
    )
    .expect("off load");
    assert_eq!(engine_rope_scaling(&off), RopeScaling::None);

    // The scope ended with the constructor: a plain load is unaffected.
    let plain = InferenceEngine::from_gguf_with_backend(
        &gguf,
        SamplingParams::default(),
        42,
        64,
        crate::engine_seam::Backend::Cpu,
    )
    .expect("plain load");
    assert!(matches!(
        engine_rope_scaling(&plain),
        RopeScaling::Yarn { .. }
    ));
}

#[test]
// `expect_err` needs `T: Debug`; `InferenceEngine` does not implement it
// (by design — it can hold multi-GB weights), so this reaches the same
// outcome through `.err().expect(..)` instead.
#[allow(clippy::err_expect)]
fn rope_override_on_refuses_a_gguf_without_scaling() {
    use oxibonsai_core::config::RopeScalingOverride;
    let bytes = build_tiny_gguf_bytes();
    let gguf = oxibonsai_core::gguf::reader::GgufFile::parse(&bytes).expect("parse fixture");
    let err = InferenceEngine::from_gguf_with_backend_and_rope(
        &gguf,
        SamplingParams::default(),
        42,
        64,
        crate::engine_seam::Backend::Cpu,
        RopeScalingOverride::On,
    )
    .err()
    .expect("`on` must refuse a file that declares no scaling");
    assert!(err.to_string().contains("--rope-scaling on"), "{err}");

    let at_load =
        crate::engine_seam::resolve_rope_scaling_at_load(&gguf, RopeScalingOverride::Auto)
            .expect("auto resolves");
    assert_eq!(at_load.declared, oxibonsai_core::config::RopeScaling::None);
    assert_eq!(at_load.effective, oxibonsai_core::config::RopeScaling::None);
}

/// Every replica of a pool built with `off` gets plain RoPE (all of them
/// are constructed inside the one scope, on this thread).
#[tokio::test]
async fn rope_override_off_reaches_every_pool_replica() {
    use oxibonsai_core::config::{RopeScaling, RopeScalingOverride};
    let bytes: &'static [u8] =
        Box::leak(build_tiny_gguf_bytes_with(yarn_metadata()).into_boxed_slice());
    let gguf: &'static oxibonsai_core::gguf::reader::GgufFile<'static> = Box::leak(Box::new(
        oxibonsai_core::gguf::reader::GgufFile::parse(bytes).expect("parse fixture"),
    ));
    let built = build_pool_from_static_gguf_with_rope(
        gguf,
        SamplingParams::default(),
        42,
        64,
        Some(2),
        crate::engine_seam::Backend::Cpu,
        RopeScalingOverride::Off,
    )
    .expect("build pool");
    assert_eq!(built.size, 2);
    let first = built.pool.acquire().await.expect("replica 1");
    let second = built.pool.acquire().await.expect("replica 2");
    assert_eq!(engine_rope_scaling(&first), RopeScaling::None);
    assert_eq!(engine_rope_scaling(&second), RopeScaling::None);
}

/// `GpuSession::for_tier(KernelTier::Gpu)` twice hands out two **real,
/// distinct** sessions.
/// `every_replica_gets_its_own_gpu_session` runs on a CPU-tier fixture,
/// where both ids are `None` and its distinctness check is vacuous; this
/// calls the constructor on the GPU tier directly. A CPU tier still gets
/// no session. Self-skips with a capability record when the host has no
/// Metal device.
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn gpu_tier_sessions_are_real_and_distinct() {
    use oxibonsai_kernels::KernelTier;
    use oxibonsai_testkit::capability::{record_executed_timed, record_skipped, Capability};

    const TEST: &str = "engine_pool::tests::gpu_tier_sessions_are_real_and_distinct";
    if let Err(e) = oxibonsai_kernels::MetalGraph::new_session() {
        eprintln!("capability report: {TEST} SKIPPED -- no Metal device ({e})");
        record_skipped(Capability::Metal, TEST);
        return;
    }
    let gate_start = std::time::Instant::now();
    let a = GpuSession::for_tier(KernelTier::Gpu);
    let b = GpuSession::for_tier(KernelTier::Gpu);
    match (a.id(), b.id()) {
        (Some(x), Some(y)) => assert_ne!(x, y, "two GPU-tier replicas share one Metal session"),
        other => {
            panic!("a GPU-tier replica on a Metal host must get its own session: {other:?}")
        }
    }
    assert_eq!(
        GpuSession::for_tier(KernelTier::Reference).id(),
        None,
        "a CPU-tier replica never opens a session"
    );
    record_executed_timed(Capability::Metal, TEST, gate_start.elapsed());
}

/// A GPU build without the Metal backend (`native-cuda`, or `metal` off
/// macOS): the GPU tier exists, but there is no Metal session to hand
/// out, so the placeholder reports none on the GPU tier and on a CPU one.
///
/// Gated on the runtime features that turn on `oxibonsai-kernels/gpu` —
/// the only builds in which `KernelTier::Gpu` exists (the runtime crate
/// has no `gpu` feature of its own).
#[cfg(all(
    any(feature = "metal", feature = "native-cuda"),
    not(all(feature = "metal", target_os = "macos"))
))]
#[test]
fn gpu_tier_sessions_are_real_and_distinct() {
    use oxibonsai_kernels::KernelTier;
    use oxibonsai_testkit::capability::{record_skipped, Capability};

    const TEST: &str = "engine_pool::tests::gpu_tier_sessions_are_real_and_distinct";
    assert_eq!(GpuSession::for_tier(KernelTier::Gpu).id(), None);
    assert_eq!(GpuSession::for_tier(KernelTier::Reference).id(), None);
    eprintln!("capability report: {TEST} SKIPPED -- built without the Metal backend");
    record_skipped(Capability::Metal, TEST);
}

/// A build with no GPU backend at all: `KernelTier::Gpu` is not even
/// compiled (it needs `oxibonsai-kernels/gpu`), so there is no GPU-tier
/// session to test; a CPU tier still gets none.
#[cfg(not(any(feature = "metal", feature = "native-cuda")))]
#[test]
fn gpu_tier_sessions_are_real_and_distinct() {
    use oxibonsai_testkit::capability::{record_skipped, Capability};

    const TEST: &str = "engine_pool::tests::gpu_tier_sessions_are_real_and_distinct";
    assert_eq!(
        GpuSession::for_tier(oxibonsai_kernels::KernelTier::Reference).id(),
        None
    );
    eprintln!("capability report: {TEST} SKIPPED -- built without a GPU backend");
    record_skipped(Capability::Metal, TEST);
}

// ── EnginePool::set_min_p_all ────────────────────────────────────────────

#[tokio::test]
async fn set_min_p_all_reaches_every_idle_replica() {
    let pool = EnginePool::new(vec![tiny_engine(), tiny_engine()]);
    pool.set_min_p_all(0.37).expect("set_min_p_all");
    for _ in 0..pool.size() {
        let lease = pool.acquire().await.expect("acquire");
        assert!(
            (lease.min_p() - 0.37).abs() < f32::EPSILON,
            "min_p was not applied to every replica: got {}",
            lease.min_p()
        );
    }
}

// ── pool-builder dedupe ──────────────────────────────────────────────────

/// `build_pool_from_gguf_parts` and `build_pool_from_gguf_parts_with_rope`
/// (which defers to `build_pool_from_static_gguf_with_rope`) now share one
/// private `finish_pool_from_first_replica` for everything past replica
/// #1's own construction. Both builders on the SAME tiny GGUF must
/// therefore still produce identical replica counts and tier reports.
#[test]
fn both_pool_builders_agree_on_size_and_tier_for_the_same_gguf() {
    let bytes = build_tiny_gguf_bytes();
    let path = {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "oxibonsai_pool_builder_dedupe_{}_{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        p
    };
    std::fs::write(&path, &bytes).expect("write temp GGUF");

    let plain = build_pool_from_gguf_parts(
        &path,
        SamplingParams::default(),
        42,
        512,
        Some(2),
        crate::engine_seam::Backend::Auto,
    )
    .expect("build_pool_from_gguf_parts");
    let with_rope = build_pool_from_gguf_parts_with_rope(
        &path,
        SamplingParams::default(),
        42,
        512,
        Some(2),
        crate::engine_seam::Backend::Auto,
        oxibonsai_core::config::RopeScalingOverride::Auto,
    )
    .expect("build_pool_from_gguf_parts_with_rope");

    let _ = std::fs::remove_file(&path);

    assert_eq!(plain.size, with_rope.size, "replica counts must agree");
    assert_eq!(
        plain.tier, with_rope.tier,
        "resolved kernel tiers must agree"
    );
    assert_eq!(plain.hybrid, with_rope.hybrid);
}
