//! Load-robust speed benchmarks for the f32-exact Metal TE GEMM
//! (`encode_gemm_f32`, wired via [`oxibonsai_image::te::gpu::te_matmul_gpu`])
//! vs the CPU [`oxibonsai_image::gemm::gemm_abt`].
//!
//! Both are gated on `cfg(all(feature = "metal", target_os = "macos"))` and
//! skip cleanly when no Metal device is present.
//! [`bench_te_gemm_f32_cpu_vs_gpu_ratio`] is a wall-clock timing comparison
//! and stays `#[ignore]`d (run explicitly, never part of the default gate);
//! [`bench_te_forward_cold_vs_warm`] is a real-model gate — it self-skips
//! with a capability record when the TE weights are absent instead, and is
//! not `#[ignore]`d.
//!
//! Run:
//! ```text
//! rustup run nightly cargo test -p oxibonsai-image --features metal --release \
//!     --test te_gpu_bench -- --ignored --nocapture
//! ```

#![cfg(all(feature = "metal", target_os = "macos"))]

use std::path::PathBuf;
use std::time::Instant;

use oxibonsai_image::gemm::gemm_abt;
use oxibonsai_image::te::gpu::{te_gpu_enabled, te_gpu_was_used, te_matmul_gpu};
use oxibonsai_image::te::{TeWeights, TextEncoder};
use oxibonsai_testkit::capability::{record_executed_timed, record_skipped, Capability};

/// Build a deterministic row-major f32 weight `[n, k]` (bounded values).
fn build_weight(n: usize, k: usize) -> Vec<f32> {
    let mut w = vec![0f32; n * k];
    let mut lcg: u32 = 0x2545_F491 ^ ((n as u32) << 5) ^ (k as u32).wrapping_mul(2_246_822_519);
    for v in w.iter_mut() {
        lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *v = ((lcg >> 8) as f32 / (1u32 << 24) as f32) - 0.5;
    }
    w
}

/// One benchmark case: `(m, n, k, weight[n*k], input[m*k])`, all kept alive
/// together so `te_matmul_gpu`'s pointer-keyed weight cache never collides.
type BenchCase = (usize, usize, usize, Vec<f32>, Vec<f32>);

/// Median of an odd-length timing series (ms).
fn median(mut t: Vec<f64>) -> f64 {
    t.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    t[t.len() / 2]
}

/// Back-to-back CPU `gemm_abt` vs GPU `encode_gemm_f32` ratio on the two
/// representative TE matmul shapes (M=512, N=9728, K=2560 and M=512, N=2560,
/// K=9728), warm cache, median of ≥ 9 iterations. Interleaved so both see the
/// same machine load. Also asserts parity (max-abs < 1e-3) so the timing isn't
/// comparing against a broken result.
#[test]
#[ignore = "GPU benchmark — run explicitly with --ignored --nocapture"]
fn bench_te_gemm_f32_cpu_vs_gpu_ratio() {
    // Probe: skip cleanly if no Metal device / GPU path unusable.
    {
        let w = build_weight(64, 256);
        let x = vec![0.01f32; 256];
        let mut o = vec![0f32; 64];
        // Weight is alive for the probe duration → stable pointer → resident=true.
        if te_matmul_gpu(&w, &x, &mut o, 1, 64, 256, true).is_err() {
            eprintln!("no usable Metal GPU — skipping TE f32 GEMM ratio bench");
            return;
        }
    }

    // (M, N, K) — the two spec shapes (down_proj-ish and up/gate-ish).
    let shapes = [(512usize, 9728usize, 2560usize), (512, 2560, 9728)];
    const WARMUP: usize = 3;
    const ITERS: usize = 9;

    // Build every weight + input UP FRONT and keep them alive for the whole
    // test. `te_matmul_gpu` caches the GPU buffer by `weight.as_ptr()`; if a
    // weight Vec were freed before the next shape allocated, the allocator could
    // hand back the same address → a stale cache hit of the wrong matrix. (The
    // real TE forward never does this: all 252 weights are distinct live
    // allocations in `TeWeights` for the entire forward.) Holding them all live
    // guarantees distinct, non-colliding pointer keys.
    let data: Vec<BenchCase> = shapes
        .iter()
        .map(|&(m, n, k)| {
            let weight = build_weight(n, k);
            let input: Vec<f32> = (0..m * k)
                .map(|i| ((i % 251) as f32) * 0.001 - 0.12)
                .collect();
            (m, n, k, weight, input)
        })
        .collect();

    for (m, n, k, weight, input) in &data {
        let (m, n, k) = (*m, *n, *k);

        // Parity guard: GPU vs CPU on this exact shape.
        // Weights are alive for the whole bench loop → stable pointers → resident=true.
        let mut gpu_out = vec![0f32; m * n];
        te_matmul_gpu(weight, input, &mut gpu_out, m, n, k, true).expect("GPU matmul failed");
        let mut cpu_out = vec![0f32; m * n];
        gemm_abt(input, weight, &mut cpu_out, m, n, k);
        let max_abs = gpu_out
            .iter()
            .zip(cpu_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_abs < 1e-3,
            "M={m} N={n} K={k}: GPU/CPU max-abs {max_abs:e} exceeds 1e-3"
        );

        let gpu_once = || {
            let mut out = vec![0f32; m * n];
            // Weights are alive for the bench loop → stable pointers → resident=true.
            te_matmul_gpu(weight, input, &mut out, m, n, k, true).expect("GPU matmul failed");
        };
        let cpu_once = || {
            let mut out = vec![0f32; m * n];
            gemm_abt(input, weight, &mut out, m, n, k);
        };

        let measure = |mut f: Box<dyn FnMut()>| -> f64 {
            for _ in 0..WARMUP {
                f();
            }
            let mut times = Vec::with_capacity(ITERS);
            for _ in 0..ITERS {
                let t0 = Instant::now();
                f();
                times.push(t0.elapsed().as_secs_f64() * 1e3);
            }
            median(times)
        };

        // Interleave to share machine state.
        let gpu_a = measure(Box::new(gpu_once));
        let cpu_a = measure(Box::new(cpu_once));
        let gpu_b = measure(Box::new(gpu_once));
        let cpu_b = measure(Box::new(cpu_once));
        let gpu_med = (gpu_a + gpu_b) / 2.0;
        let cpu_med = (cpu_a + cpu_b) / 2.0;
        let gflop = 2.0 * m as f64 * n as f64 * k as f64 / 1e9;

        eprintln!(
            "TE-GEMM M={m} N={n} K={k}: CPU={cpu_med:.3}ms ({:.1} GFLOP/s)  GPU={gpu_med:.3}ms ({:.1} GFLOP/s)  \
             speedup CPU/GPU={:.2}x  (parity max-abs={max_abs:e}, median of {ITERS}, warmup {WARMUP}, 2 reps)",
            gflop / (cpu_med / 1e3),
            gflop / (gpu_med / 1e3),
            cpu_med / gpu_med,
        );
    }
    // Confirm the GPU path engaged.
    assert!(te_gpu_was_used(), "te_gpu_was_used() == false after bench");
}

/// End-to-end TE forward wall-time, isolating the one-time weight upload.
///
/// Loads the real TE weights once and runs the full Qwen3-4B encoder forward
/// **twice**, reporting the cold forward (first) and the warm forward
/// (second). The [`TeWeights::open_mlx_4bit`] source only *caches* the
/// dequantised weights across forwards once `OXI_TE_RESIDENT=1` turns on
/// [`TeWeights::set_resident`] (its documented opt-in for a high-memory
/// machine); left unset, [`TeWeights::get`] re-dequantises every tensor on
/// every forward by design (the low-RAM default), so cold and warm report
/// the same cost — a real, not a missing-warm-up, result. With
/// `OXI_TE_RESIDENT=1`, the warm forward measures the production-relevant
/// cost of a resident session instead. With `OXI_TE_GPU=1` this also shows
/// GPU wall-time; without it (CPU), compare across two process runs for
/// CPU-vs-GPU.
///
/// Resolves the native ~2.1 GB 4-bit MLX safetensors from `OXI_TE_4BIT`
/// (the same variable `examples/generate.rs` and `examples/te_parity.rs`
/// check, via [`TeWeights::open_mlx_4bit`]) and self-skips, recording the
/// miss under `Capability::ImageParity`, when it is unset — this
/// deliberately never falls back to the legacy ~15/16 GB f32 `.npy` dump
/// directory `TeWeights::open` reads — loading that much is exactly what
/// resolving through the native 4-bit source instead of the legacy one
/// avoids.
#[test]
fn bench_te_forward_cold_vs_warm() {
    const TEST: &str = "oxibonsai-image::te_gpu_bench::bench_te_forward_cold_vs_warm";
    let Some(safetensors_path) = std::env::var("OXI_TE_4BIT")
        .ok()
        .filter(|p| !p.is_empty())
        .map(PathBuf::from)
    else {
        eprintln!(
            "OXI_TE_4BIT not set — skipping end-to-end wall-time bench (point it at the native \
             4-bit text_encoder-mlx-4bit/model.safetensors to run)"
        );
        record_skipped(Capability::ImageParity, TEST);
        return;
    };
    if !safetensors_path.is_file() {
        eprintln!(
            "OXI_TE_4BIT names {} which is not a file — skipping",
            safetensors_path.display()
        );
        record_skipped(Capability::ImageParity, TEST);
        return;
    }
    let start = Instant::now();
    let weights = match TeWeights::open_mlx_4bit(&safetensors_path) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("open TE weights failed: {e} — skipping");
            record_skipped(Capability::ImageParity, TEST);
            return;
        }
    };
    let resident = std::env::var("OXI_TE_RESIDENT")
        .ok()
        .is_some_and(|v| v == "1");
    weights.set_resident(resident);
    let encoder = TextEncoder::new(&weights);

    // Short prompt-like input (causal, all real tokens). 32 tokens keeps the
    // attention/norm CPU cost modest so the matmul share dominates.
    let seq = 32usize;
    let input_ids: Vec<u32> = (0..seq).map(|i| (1000 + i * 7) as u32 % 150_000).collect();
    let attention_mask: Vec<i32> = vec![1; seq];

    let gpu = te_gpu_enabled();
    eprintln!(
        "TE end-to-end (seq={seq}, OXI_TE_GPU={}, OXI_TE_RESIDENT={}):",
        if gpu { "1 (GPU)" } else { "0 (CPU)" },
        i32::from(resident)
    );

    let t0 = Instant::now();
    let _o1 = encoder
        .forward(&input_ids, &attention_mask)
        .expect("forward 1");
    let cold = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let _o2 = encoder
        .forward(&input_ids, &attention_mask)
        .expect("forward 2");
    let warm = t1.elapsed().as_secs_f64();

    eprintln!(
        "  cold forward = {cold:.2}s   warm forward = {warm:.2}s   (cold includes one-time weight \
         upload when GPU; cold ~= warm is expected with OXI_TE_RESIDENT unset — every tensor is \
         re-dequantised on both forwards by design)"
    );
    if gpu {
        assert!(
            te_gpu_was_used(),
            "OXI_TE_GPU=1 but te_gpu_was_used() == false (CPU fallback)"
        );
    }
    record_executed_timed(Capability::ImageParity, TEST, start.elapsed());
}
