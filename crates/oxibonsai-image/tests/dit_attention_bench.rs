//! HONEST CPU-vs-GPU benchmark for the FLUX.2 DiT joint-attention at the real
//! DiT shape `(num_heads=24, seq=1536, head_dim=128)`.
//!
//! Compares the **real** rayon+NEON CPU path
//! [`oxibonsai_image::math::joint_attention`] (the actual production code — NOT a
//! scalar port) against the Metal flash-attention `joint_attention_flash_f32`
//! kernel (the shipping DiT attention path) in two regimes:
//!
//! 1. **CPU real** — `math::joint_attention` (rayon over 24 heads, NEON
//!    `gemm::dot`, `softmax_simd`).
//! 2. **GPU kernel-only** — q/k/v uploaded to resident GPU buffers ONCE before
//!    the loop; per-iter timing covers only encode + commit + wait (no upload,
//!    no download). This is the **resident-scenario** number — the upper bound
//!    on what a fused-resident DiT could achieve for attention compute alone.
//! 3. **GPU with transfers** — per-iter upload(q,k,v) + encode + wait +
//!    download(out), using pooled (reused) buffers so the GPU is not penalised
//!    by per-call allocation. This is the **standalone-scenario** number.
//!
//! Reports all three in ms + the two ratios (CPU/GPU-kernel-only and
//! CPU/GPU-with-transfers), plus the machine load average (via `uptime`).
//!
//! `#[ignore]` (run explicitly) and gated on
//! `cfg(all(feature = "metal", target_os = "macos"))`; skips cleanly when no
//! Metal device is present.
//!
//! Run:
//! ```text
//! rustup run nightly cargo test -p oxibonsai-image --features metal --release \
//!     --test dit_attention_bench -- --ignored --nocapture
//! ```

#![cfg(all(feature = "metal", target_os = "macos"))]

use std::time::Instant;

use oxibonsai_image::math::joint_attention;
use oxibonsai_kernels::MetalGraph;

/// Deterministic pseudo-random fill in roughly `[-1, 1]` from an integer seed
/// (LCG; stable across platforms — same generator as the kernel parity test).
fn fill_deterministic(buf: &mut [f32], seed_base: u64) {
    let mut state = seed_base.wrapping_mul(6364136223846793005).wrapping_add(1);
    for x in buf.iter_mut() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (state >> 33) as u32;
        let unit = (bits as f32) / (u32::MAX as f32);
        *x = unit * 2.0 - 1.0;
    }
}

/// Median of a timing series (ms).
fn median(mut t: Vec<f64>) -> f64 {
    t.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    t[t.len() / 2]
}

/// Max-abs difference between two equal-length slices.
fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

/// Best-effort machine load average string via `uptime` (macOS has no
/// `/proc/loadavg`). Returns an empty string if the command is unavailable.
fn loadavg_str() -> String {
    std::process::Command::new("uptime")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

/// HONEST three-number benchmark at the DiT shape (24, 1536, 128).
#[test]
#[ignore = "GPU benchmark — run explicitly with --ignored --nocapture"]
fn bench_joint_attention_cpu_vs_gpu() {
    // Probe: skip cleanly if no usable Metal GPU.
    let graph = match MetalGraph::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no usable Metal GPU ({e}) — skipping DiT joint-attention bench");
            return;
        }
    };

    let (num_heads, seq, head_dim) = (24usize, 1536usize, 128usize);
    let qkv_len = num_heads * seq * head_dim;
    let out_len = seq * num_heads * head_dim;

    let mut q = vec![0.0f32; qkv_len];
    let mut k = vec![0.0f32; qkv_len];
    let mut v = vec![0.0f32; qkv_len];
    fill_deterministic(&mut q, 0xA1);
    fill_deterministic(&mut k, 0xB2);
    fill_deterministic(&mut v, 0xC3);

    // ── Parity guard (don't time a broken result) ───────────────────────────
    let cpu_ref =
        joint_attention(&q, &k, &v, num_heads, seq, head_dim).expect("CPU joint_attention failed");
    let mut gpu_check = vec![0.0f32; out_len];
    graph
        .encode_joint_attention_flash_pooled(&q, &k, &v, &mut gpu_check, num_heads, seq, head_dim)
        .expect("GPU joint_attention (flash pooled) failed");
    let parity = max_abs(&cpu_ref, &gpu_check);
    assert!(
        parity < 1e-3,
        "DiT joint-attention GPU/CPU max-abs {parity:e} exceeds 1e-3 — refusing to benchmark"
    );

    const WARMUP: usize = 3;
    const ITERS: usize = 15; // odd, >= 11

    // (2) GPU kernel-only (resident): prepare uploads q/k/v into the resident
    // pool exactly ONCE here; the timed dispatch closure (below) reuses those
    // already-resident buffers and leaves the output on the GPU (no per-iter
    // upload/download).
    graph
        .joint_attn_resident_prepare(&q, &k, &v, &cpu_ref, num_heads, seq, head_dim)
        .expect("joint_attn_resident_prepare failed");

    // (3) GPU with transfers: reusable output buffer hoisted so its allocation
    // is *not* in the timed region (the pooled encode reuses GPU-side buffers).
    let mut gpu_out = vec![0.0f32; out_len];

    fn measure<F: FnMut()>(warmup: usize, iters: usize, mut f: F) -> f64 {
        for _ in 0..warmup {
            f();
        }
        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            f();
            times.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        median(times)
    }

    // Fresh inline closures per rep (each captures only shared refs / the local
    // `gpu_out`, so recreating them is zero-cost and avoids move/borrow clashes).
    // Interleaved so each path sees comparable machine state (two reps each).
    //   (1) CPU real (rayon+NEON)  (2) GPU kernel-only  (3) GPU with transfers.
    let cpu_a = measure(WARMUP, ITERS, || {
        let r = joint_attention(&q, &k, &v, num_heads, seq, head_dim)
            .expect("CPU joint_attention failed");
        std::hint::black_box(&r);
    });
    let gk_a = measure(WARMUP, ITERS, || {
        graph
            .joint_attn_flash_resident_dispatch(num_heads, seq, head_dim)
            .expect("joint_attn_flash_resident_dispatch failed");
    });
    let gx_a = measure(WARMUP, ITERS, || {
        graph
            .encode_joint_attention_flash_pooled(&q, &k, &v, &mut gpu_out, num_heads, seq, head_dim)
            .expect("encode_joint_attention_flash_pooled failed");
        std::hint::black_box(&gpu_out);
    });
    let cpu_b = measure(WARMUP, ITERS, || {
        let r = joint_attention(&q, &k, &v, num_heads, seq, head_dim)
            .expect("CPU joint_attention failed");
        std::hint::black_box(&r);
    });
    let gk_b = measure(WARMUP, ITERS, || {
        graph
            .joint_attn_flash_resident_dispatch(num_heads, seq, head_dim)
            .expect("joint_attn_flash_resident_dispatch failed");
    });
    let gx_b = measure(WARMUP, ITERS, || {
        graph
            .encode_joint_attention_flash_pooled(&q, &k, &v, &mut gpu_out, num_heads, seq, head_dim)
            .expect("encode_joint_attention_flash_pooled failed");
        std::hint::black_box(&gpu_out);
    });

    let cpu_ms = (cpu_a + cpu_b) / 2.0;
    let gpu_kernel_ms = (gk_a + gk_b) / 2.0;
    let gpu_xfer_ms = (gx_a + gx_b) / 2.0;

    let ratio_kernel = cpu_ms / gpu_kernel_ms;
    let ratio_xfer = cpu_ms / gpu_xfer_ms;
    let load = loadavg_str();

    eprintln!("\n══════════════════════════════════════════════════════════════════════");
    eprintln!(
        "DiT joint-attention (heads={num_heads}, seq={seq}, head_dim={head_dim}); \
         parity max-abs={parity:e}; median of {ITERS}, warmup {WARMUP}, 2 reps interleaved"
    );
    eprintln!("  (1) CPU real (rayon+NEON)        = {cpu_ms:8.3} ms");
    eprintln!(
        "  (2) GPU kernel-only (resident)   = {gpu_kernel_ms:8.3} ms   \
         [CPU/GPU = {ratio_kernel:.2}x]"
    );
    eprintln!(
        "  (3) GPU with transfers (pooled)  = {gpu_xfer_ms:8.3} ms   \
         [CPU/GPU = {ratio_xfer:.2}x]"
    );
    if !load.is_empty() {
        eprintln!("  loadavg: {load}");
    }
    eprintln!("══════════════════════════════════════════════════════════════════════\n");

    // Informational only — no perf assertion (this benchmark *informs* a
    // go/no-go decision; it must not fail CI on a loaded machine). The parity
    // gate above is the only hard assertion.
}
