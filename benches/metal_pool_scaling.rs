//! Metal engine-pool serving throughput vs. pool size (`MET-08`).
//!
//! Builds GPU engine pools of 1, 2 and 3 replicas over one real ternary GGUF
//! and measures how long each takes to serve the same batch of 8 concurrent
//! short greedy requests — acquired on the async runtime and generated inside
//! `spawn_blocking`, exactly as the server does. Criterion reports the batch
//! time and, through `Throughput::Elements`, generated tokens per second.
//!
//! The model comes from `OXI_MODEL` (e.g. `Ternary-Bonsai-1.7B.gguf`); without
//! it — or off the Metal GPU tier — the bench prints why and measures nothing.
//! The relative-invariant twin that runs in the test gate is
//! `crates/oxibonsai-runtime/tests/metal_concurrency_tests.rs`
//! (`metal_concurrency_real_model_pool_scaling_and_byte_identity`).
//!
//! Run with:
//! ```text
//! OXI_MODEL=<path>/Ternary-Bonsai-1.7B.gguf \
//!   cargo bench --features bench,metal --bench metal_pool_scaling
//! ```

use criterion::{criterion_group, criterion_main, Criterion};

// Everything below is only used by the measuring path, which exists on macOS
// with the `metal` feature (the `mod pool` and `bench_pool_scaling` arms below
// carry the same predicate). Everywhere else the bench just prints why it
// measured nothing, so these must not be declared there or they are unused.
#[cfg(all(feature = "metal", target_os = "macos"))]
use criterion::{BenchmarkId, Throughput};
#[cfg(all(feature = "metal", target_os = "macos"))]
use std::sync::Arc;
#[cfg(all(feature = "metal", target_os = "macos"))]
use std::time::Duration;

/// Concurrent requests per measured batch.
#[cfg(all(feature = "metal", target_os = "macos"))]
const REQUESTS: usize = 8;
/// Tokens generated per request.
#[cfg(all(feature = "metal", target_os = "macos"))]
const MAX_TOKENS: usize = 24;
/// Context each replica is built with.
#[cfg(all(feature = "metal", target_os = "macos"))]
const MAX_SEQ: usize = 256;

#[cfg(all(feature = "metal", target_os = "macos"))]
mod pool {
    use super::{MAX_SEQ, MAX_TOKENS, REQUESTS};
    use oxibonsai_kernels::KernelTier;
    use oxibonsai_runtime::engine_pool::{build_pool_from_gguf, EnginePool};
    use oxibonsai_runtime::sampling::SamplingParams;
    use std::sync::Arc;

    /// Greedy decoding (the fused GPU-argmax route).
    pub fn greedy() -> SamplingParams {
        SamplingParams {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 1.0,
            max_tokens: MAX_TOKENS,
        }
    }

    /// The batch every pool serves: 8 short prompts.
    pub fn prompts() -> Vec<Vec<u32>> {
        (0..REQUESTS)
            .map(|r| (0..6).map(|i| (1000 + r * 977 + i * 131) as u32).collect())
            .collect()
    }

    /// A GPU pool of `replicas` over `path`, or why it could not be built.
    pub fn build(path: &std::ffi::OsStr, replicas: usize) -> Result<Arc<EnginePool>, String> {
        let (pool, tier, size) = build_pool_from_gguf(path, greedy(), 42, MAX_SEQ, Some(replicas))
            .map_err(|e| format!("build_pool_from_gguf: {e}"))?;
        if tier != KernelTier::Gpu {
            return Err(format!(
                "the model resolved to {tier}, not the Metal GPU tier"
            ));
        }
        if size != replicas {
            return Err(format!("asked for {replicas} replicas, got {size}"));
        }
        Ok(pool)
    }

    /// Serve every prompt concurrently; returns the tokens generated.
    pub async fn serve(pool: &Arc<EnginePool>, prompts: &[Vec<u32>]) -> Result<usize, String> {
        let mut handles = Vec::with_capacity(prompts.len());
        for prompt in prompts {
            let pool = Arc::clone(pool);
            let prompt = prompt.clone();
            handles.push(tokio::spawn(async move {
                let lease = pool.acquire().await.map_err(|e| e.to_string())?;
                tokio::task::spawn_blocking(move || {
                    let mut lease = lease;
                    lease
                        .generate(&prompt, MAX_TOKENS)
                        .map(|tokens| tokens.len())
                        .map_err(|e| e.to_string())
                })
                .await
                .map_err(|e| e.to_string())?
            }));
        }
        let mut total = 0usize;
        for handle in handles {
            total += handle.await.map_err(|e| e.to_string())??;
        }
        Ok(total)
    }
}

/// Measure batch-serving time for pools of 1, 2 and 3 replicas.
fn bench_pool_scaling(c: &mut Criterion) {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        let Some(path) = std::env::var_os("OXI_MODEL") else {
            eprintln!("metal_pool_scaling: OXI_MODEL not set — nothing to measure");
            return;
        };
        let runtime = match tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .build()
        {
            Ok(runtime) => runtime,
            Err(e) => {
                eprintln!("metal_pool_scaling: tokio runtime: {e}");
                return;
            }
        };
        let prompts = pool::prompts();
        let mut group = c.benchmark_group("metal_pool_scaling");
        group.sample_size(10);
        group.warm_up_time(Duration::from_secs(5));
        group.measurement_time(Duration::from_secs(45));
        for replicas in 1..=3usize {
            let pool = match pool::build(&path, replicas) {
                Ok(pool) => pool,
                Err(why) => {
                    eprintln!("metal_pool_scaling: pool of {replicas}: {why} — skipping");
                    continue;
                }
            };
            // Warm every replica (weights upload, device KV allocation) and
            // learn the batch's token count for the throughput line.
            let tokens = match runtime.block_on(pool::serve(&pool, &prompts)) {
                Ok(tokens) => tokens,
                Err(why) => {
                    eprintln!("metal_pool_scaling: pool of {replicas}: warm-up failed: {why}");
                    continue;
                }
            };
            group.throughput(Throughput::Elements(tokens as u64));
            group.bench_with_input(
                BenchmarkId::new("8_concurrent_greedy_requests", replicas),
                &Arc::clone(&pool),
                |b, pool| {
                    b.iter(|| {
                        runtime
                            .block_on(pool::serve(pool, &prompts))
                            .map_err(|why| eprintln!("metal_pool_scaling: {why}"))
                            .ok()
                    })
                },
            );
        }
        group.finish();
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        let _ = c;
        eprintln!("metal_pool_scaling: built without the Metal backend — nothing to measure");
    }
}

criterion_group!(benches, bench_pool_scaling);
criterion_main!(benches);
