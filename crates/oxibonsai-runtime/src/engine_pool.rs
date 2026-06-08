//! Engine-replica pool for concurrent serving.
//!
//! The HTTP server historically wrapped a single [`InferenceEngine`] in a
//! `tokio::sync::Mutex`, which serialized every request: a long generation
//! blocked all others for its full duration. This module replaces that single
//! mutex with a *pool* of `N` engine replicas guarded by a semaphore, so up to
//! `N` requests can generate concurrently.
//!
//! ## Why a pool works (and is safe)
//!
//! - Weights are process-global and immutable: [`InferenceEngine::from_gguf_path`]
//!   leaks the mmap and parsed `GgufFile` to `'static`, so every replica borrows
//!   the *same* `&'static GgufFile` zero-copy. The immutable, dequantized
//!   `token_embd` table is likewise shared across replicas via one `Arc<[f32]>`
//!   (see [`build_pool_from_gguf`]). Only the per-replica `KvCache` and light
//!   wrappers are duplicated.
//! - On CPU tiers (Reference / AVX / NEON), `BonsaiModel::forward` mutates only
//!   `self.kv_cache` over a shared `&dyn OneBitKernel` on immutable weights, so
//!   distinct engine instances run fully parallel.
//! - On the GPU tier (Metal / CUDA), decode funnels through a process-global
//!   singleton graph owning one KV cache and shared scratch. `N > 1` GPU
//!   replicas give *no* compute parallelism and would corrupt each other's KV,
//!   so the pool size is clamped to `1` on the GPU tier (see
//!   [`resolve_pool_size`]).
//!
//! ## Back-compatibility
//!
//! The default path wraps exactly one engine in a 1-element pool whose
//! [`EngineLease`] calls the identical `generate*` methods on the identical
//! engine. Single-request behavior — including RNG progression — is therefore
//! byte-identical to the previous single-mutex design.
//!
//! ## Lock discipline
//!
//! `idle` is a *synchronous* [`std::sync::Mutex`] held only for the duration of
//! a `pop`/`push` — never across an `.await`. Async waiting happens purely on
//! the [`tokio::sync::Semaphore`], whose permit count equals the pool size.

use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::engine::InferenceEngine;

/// Errors that can occur while acquiring an engine from the pool.
///
/// These map to HTTP `503 Service Unavailable` at the call site — they all
/// represent a transient inability to serve, never a client error.
#[derive(Debug, thiserror::Error)]
pub enum PoolError {
    /// The pool's semaphore was closed (the pool is shutting down).
    #[error("engine pool is closed")]
    Closed,
    /// A permit was acquired but no idle engine was available.
    ///
    /// This cannot happen under correct operation — permits and idle engines
    /// are kept in lock-step by construction — but is surfaced as an error
    /// rather than a panic to uphold the crate's no-panic policy.
    #[error("engine pool is unexpectedly empty")]
    Empty,
    /// The `idle` mutex was poisoned by a panic in another thread.
    #[error("engine pool mutex was poisoned")]
    Poisoned,
}

/// A pool of [`InferenceEngine`] replicas guarded by a semaphore.
///
/// Construct with [`EnginePool::new`]; acquire an engine with
/// [`EnginePool::acquire`]. The returned [`EngineLease`] derefs to the engine
/// and returns it to the pool on drop.
pub struct EnginePool {
    /// Idle (available) engines. Guarded by a *synchronous* mutex held only for
    /// `pop`/`push`. The number of engines ever simultaneously checked out plus
    /// the length of this vector always equals [`EnginePool::size`].
    idle: Mutex<Vec<InferenceEngine<'static>>>,
    /// Async gate: exactly `size` permits. A permit is held for the lifetime of
    /// each outstanding [`EngineLease`] and released only after the engine has
    /// been returned to `idle`.
    sem: Arc<Semaphore>,
    /// Number of replicas in the pool (immutable after construction).
    size: usize,
}

impl EnginePool {
    /// Build a pool from a vector of engine replicas.
    ///
    /// The pool size is `engines.len()`, guaranteed to be at least `1` (an
    /// empty input yields a 1-permit pool with no engines, which would only
    /// ever return [`PoolError::Empty`]; callers must pass at least one
    /// engine). The semaphore is seeded with `size` permits.
    pub fn new(engines: Vec<InferenceEngine<'static>>) -> Arc<Self> {
        let size = engines.len().max(1);
        Arc::new(Self {
            idle: Mutex::new(engines),
            sem: Arc::new(Semaphore::new(size)),
            size,
        })
    }

    /// Number of replicas in the pool.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Attach a shared [`crate::metrics::InferenceMetrics`] to every replica in the pool.
    ///
    /// This wires the per-engine telemetry (prefill / decode-token /
    /// tokens-per-second histograms recorded inside `generate*`) onto each
    /// replica, mirroring the single-engine `engine.set_metrics(..)` call the
    /// CLI `serve` handler performed before this pool existed. It must be called
    /// while the pool is *idle* (right after construction, before any lease is
    /// handed out), so all replicas are present in `idle`.
    ///
    /// Returns [`PoolError::Poisoned`] if the idle mutex was poisoned. The
    /// metrics `Arc` is cloned once per replica so they all share one instance.
    pub fn set_metrics_all(
        &self,
        metrics: &Arc<crate::metrics::InferenceMetrics>,
    ) -> Result<(), PoolError> {
        let mut idle = self.idle.lock().map_err(|_| PoolError::Poisoned)?;
        for engine in idle.iter_mut() {
            engine.set_metrics(Arc::clone(metrics));
        }
        Ok(())
    }

    /// Acquire an engine from the pool, waiting asynchronously if all replicas
    /// are currently in use.
    ///
    /// Returns an [`EngineLease`] that derefs to the engine and returns it to
    /// the pool on drop. The acquired permit is held for the lease's lifetime.
    pub async fn acquire(self: &Arc<Self>) -> Result<EngineLease, PoolError> {
        // Wait for a free slot. The permit count mirrors the idle count, so a
        // granted permit guarantees an idle engine is (about to be) available.
        let permit = Arc::clone(&self.sem)
            .acquire_owned()
            .await
            .map_err(|_| PoolError::Closed)?;

        // Pop an idle engine. The lock is held only for this `pop`.
        let engine = {
            let mut idle = self.idle.lock().map_err(|_| PoolError::Poisoned)?;
            idle.pop().ok_or(PoolError::Empty)?
        };

        Ok(EngineLease {
            engine: ManuallyDrop::new(engine),
            pool: Arc::clone(self),
            _permit: permit,
        })
    }
}

impl std::fmt::Debug for EnginePool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let idle_len = self.idle.lock().map(|g| g.len()).ok();
        f.debug_struct("EnginePool")
            .field("size", &self.size)
            .field("available_permits", &self.sem.available_permits())
            .field("idle_len", &idle_len)
            .finish()
    }
}

/// An exclusive lease on one engine from an [`EnginePool`].
///
/// Derefs (mutably) to the borrowed [`InferenceEngine`], so callers invoke the
/// usual `generate*` methods directly. On drop, the engine is returned to the
/// pool and the semaphore permit is released — in that order, so a waiter is
/// guaranteed to find an idle engine the instant its permit is granted.
///
/// The engine is stored in [`ManuallyDrop`] so [`Drop`] can move it back into
/// the pool without an `Option`/`unwrap` dance: this is the whole reason the
/// guard is panic-free.
pub struct EngineLease {
    engine: ManuallyDrop<InferenceEngine<'static>>,
    pool: Arc<EnginePool>,
    // Field order matters: `_permit` is declared last so it is dropped *after*
    // the explicit `Drop::drop` body below has returned the engine to `idle`.
    // (Rust drops a struct's fields in declaration order after running its
    // `Drop::drop`.) This guarantees the slot is only freed once the engine is
    // back in the pool.
    _permit: OwnedSemaphorePermit,
}

impl Deref for EngineLease {
    type Target = InferenceEngine<'static>;

    fn deref(&self) -> &Self::Target {
        // Total: `engine` is always populated until `Drop` takes it exactly
        // once, and the lease is never accessed after drop.
        &self.engine
    }
}

impl DerefMut for EngineLease {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // Total: see `deref`.
        &mut self.engine
    }
}

impl Drop for EngineLease {
    fn drop(&mut self) {
        // SAFETY: `ManuallyDrop::take` is called exactly once, here in `Drop`;
        // `self.engine` is never accessed afterwards (the struct is being
        // destroyed), so no double-take or use-after-take can occur.
        let engine = unsafe { ManuallyDrop::take(&mut self.engine) };

        // Return the engine to the pool. We deliberately do NOT run a heavy
        // reset here: every `generate*` entry point resets the model KV cache
        // at the start of the call (today's behavior), and the sampler's RNG
        // state is intentionally preserved, so resetting here would be both
        // redundant and a risk to byte-identical single-request behavior.
        match self.pool.idle.lock() {
            Ok(mut idle) => idle.push(engine),
            Err(_poisoned) => {
                // The pool mutex is poisoned (a thread panicked while holding
                // it). Dropping `engine` here is the safe choice: we must not
                // panic in `Drop`, and pushing into a poisoned pool is not
                // meaningfully recoverable. The permit still releases below.
                tracing::error!("engine pool mutex poisoned on lease return; dropping replica");
                drop(engine);
            }
        }

        // `_permit` is dropped *after* this body returns (it is the last field
        // in declaration order), so the semaphore slot frees only once the
        // engine is back in `idle`.
    }
}

/// Default pool size on CPU tiers: the host's available parallelism, capped at
/// `4`, with a floor of `1`.
///
/// The cap keeps memory bounded (each replica adds its own KV cache; the
/// `token_embd` table is shared across replicas via one `Arc<[f32]>`) while
/// still allowing a handful of concurrent generations on typical multi-core
/// hosts.
pub fn default_cpu_pool_size() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().min(4))
        .unwrap_or(1)
}

/// Resolve the effective pool size for a given kernel tier.
///
/// - On the GPU tier (Metal / CUDA), the size is forced to `1`: GPU decode
///   funnels through a process-global singleton that cannot run replicas in
///   parallel and would corrupt shared KV state. This is a correctness
///   requirement, not a tuning choice.
/// - On CPU tiers, the size is `requested` (clamped to `>= 1`) or, if `None`,
///   [`default_cpu_pool_size`].
///
/// Pure and unit-testable. The GPU comparison is gated behind the GPU-enabling
/// features (`metal` / `native-cuda`) because [`oxibonsai_kernels::KernelTier::Gpu`]
/// only exists when one of them is compiled in; non-GPU builds always take the
/// CPU branch.
pub fn resolve_pool_size(requested: Option<usize>, tier: oxibonsai_kernels::KernelTier) -> usize {
    #[cfg(any(feature = "metal", feature = "native-cuda"))]
    {
        if tier == oxibonsai_kernels::KernelTier::Gpu {
            return 1;
        }
    }
    // Silence the unused-variable lint on non-GPU builds where `tier` is not
    // inspected.
    let _ = tier;
    requested.unwrap_or_else(default_cpu_pool_size).max(1)
}

/// Build an [`EnginePool`] from a GGUF file, sizing it for the detected tier.
///
/// Replica `#1` is loaded via [`InferenceEngine::from_gguf_path`], which
/// memory-maps and parses the GGUF and leaks both to `'static`. Its kernel tier
/// is read to size the pool via [`resolve_pool_size`]; replicas `2..size` are
/// then built off the *same* leaked `&'static GgufFile` via
/// [`InferenceEngine::from_gguf_static_with_embd`], so no additional mmap or
/// weight copy occurs. Every replica is seeded identically, so the served
/// output is deterministic regardless of which replica handles a request.
///
/// ## Shared token embedding
///
/// The dequantized `token_embd` table (FP32 `vocab × hidden` — ~1.16 GiB for
/// the 1.7B) is immutable and load-once. Replica `#1` loads it into an
/// `Arc<[f32]>`; that single `Arc` is then *cloned* (a refcount bump, not a
/// data copy) into every replica `2..size`. The whole pool therefore holds
/// **one** embedding allocation regardless of `size`, rather than N duplicates,
/// and replicas `2..size` skip re-dequantizing it. Per-replica `KvCache`s stay
/// fully independent.
///
/// Returns the pool, the detected [`oxibonsai_kernels::KernelTier`], and the
/// effective size.
pub fn build_pool_from_gguf(
    path: impl AsRef<std::path::Path>,
    sampling_params: crate::sampling::SamplingParams,
    seed: u64,
    max_seq_len: usize,
    requested_size: Option<usize>,
) -> crate::error::RuntimeResult<(Arc<EnginePool>, oxibonsai_kernels::KernelTier, usize)> {
    // Replica #1 — this leaks the mmap + parsed GGUF to `'static`.
    let (first, gguf) =
        InferenceEngine::from_gguf_path_leaked(path, sampling_params.clone(), seed, max_seq_len)?;

    let tier = first.kernel_tier();
    let size = resolve_pool_size(requested_size, tier);

    if requested_size.map(|r| r > size).unwrap_or(false) {
        tracing::info!(
            requested = requested_size.unwrap_or(0),
            effective = size,
            tier = %tier,
            "engine pool size clamped to 1 on the GPU tier (process-global GPU singleton)"
        );
    } else {
        tracing::info!(size, tier = %tier, "engine pool built");
    }

    // Extract replica #1's shared token-embedding table (a cheap refcount-bumped
    // `Arc<[f32]>` handle, not a copy). Replicas 2..size clone this same `Arc`
    // instead of re-dequantizing their own ~1.16 GiB copy, so the whole pool
    // holds one embedding allocation total.
    let shared_token_embd = first.model_token_embd();

    let mut engines = Vec::with_capacity(size);
    engines.push(first);
    // Replicas 2..size reuse the already-`'static` GGUF (zero extra mmap/copy)
    // and the shared `Arc<[f32]>` token-embedding table (zero extra dequant/copy).
    for _ in 1..size {
        let replica = InferenceEngine::from_gguf_static_with_embd(
            gguf,
            sampling_params.clone(),
            seed,
            max_seq_len,
            Arc::clone(&shared_token_embd),
        )?;
        engines.push(replica);
    }

    Ok((EnginePool::new(engines), tier, size))
}

#[cfg(test)]
mod tests {
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
        use oxibonsai_core::gguf::writer::{
            GgufWriter, MetadataWriteValue, TensorEntry, TensorType,
        };

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
    fn resolve_pool_size_gpu_is_clamped_to_one() {
        // The GPU clamp is a correctness property: a process-global singleton
        // cannot run replicas in parallel.
        let tier = oxibonsai_kernels::KernelTier::Gpu;
        assert_eq!(resolve_pool_size(Some(8), tier), 1);
        assert_eq!(resolve_pool_size(None, tier), 1);
        assert_eq!(resolve_pool_size(Some(1), tier), 1);
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
            .map(|l| l.model().kv_cache() as *const _)
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
}
