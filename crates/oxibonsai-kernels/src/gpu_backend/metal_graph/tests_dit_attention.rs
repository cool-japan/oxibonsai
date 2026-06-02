//! Parity tests for the FLUX.2 DiT flash-attention Metal kernel
//! (`joint_attention_flash_f32`, dispatched via
//! [`MetalGraph::encode_joint_attention_flash`] /
//! [`MetalGraph::encode_joint_attention_flash_pooled`] / the resident-buffer
//! prepare+dispatch+download flow).
//!
//! The honest CPU-vs-GPU **speed** benchmark lives in the `oxibonsai-image`
//! crate (`tests/dit_attention_bench.rs`), where the *real* rayon+NEON
//! `oxibonsai_image::math::joint_attention` is available — this module only
//! gates **correctness** (max-abs parity against a local sequential reference).
//!
//! All `#[cfg(all(feature = "metal", target_os = "macos"))]`; skip cleanly when
//! no Metal device is present.

use super::graph::MetalGraph;
use metal::Device;
use std::sync::Mutex;

/// Serializes the tests that exercise the **process-wide** joint-attention buffer
/// pool / global alloc counter (`encode_joint_attention_flash_pooled`, the
/// resident dispatch path, and `joint_attn_pool_alloc_count`). `cargo test` runs
/// tests in
/// parallel by default; without this lock the pool-grows of one test perturb the
/// `before == after` alloc-count invariant another test measures (the counter and
/// the pool are global). Correctness-only: it does not change any kernel result.
static JOINT_ATTN_POOL_TEST_SERIAL: Mutex<()> = Mutex::new(());

/// CPU reference port of `oxibonsai-image::math::joint_attention`.
///
/// Matches the reference behaviour exactly (same numerically-stabilised softmax
/// — subtract row-max, exp, normalise — sequential dot products, sequential
/// weighted-V accumulation, then the head→token transpose). Kept local to avoid
/// an `oxibonsai-image` dependency from the kernels crate.
///
/// `q`/`k`/`v` are head-major `[num_heads × seq × head_dim]`; returns the
/// token-major `[seq × (num_heads*head_dim)]` output.
fn cpu_joint_attention_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    num_heads: usize,
    seq: usize,
    head_dim: usize,
) -> Vec<f32> {
    let inner = num_heads * head_dim;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut head_out = vec![0.0f32; num_heads * seq * head_dim];

    for h in 0..num_heads {
        let head_off = h * seq * head_dim;
        let dst_base = h * seq * head_dim;
        let mut scores = vec![0.0f32; seq];
        for qi in 0..seq {
            let q_row = &q[head_off + qi * head_dim..head_off + (qi + 1) * head_dim];
            // scores[ki] = scale * dot(q_row, k_row)  (sequential over d)
            for (ki, score) in scores.iter_mut().enumerate() {
                let k_row = &k[head_off + ki * head_dim..head_off + (ki + 1) * head_dim];
                let mut acc = 0.0f32;
                for d in 0..head_dim {
                    acc += q_row[d] * k_row[d];
                }
                *score = acc * scale;
            }
            // Numerically-stable softmax: subtract row max, exp, normalise.
            let mut row_max = f32::NEG_INFINITY;
            for &s in scores.iter() {
                row_max = row_max.max(s);
            }
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - row_max).exp();
                sum += *s;
            }
            if sum != 0.0 {
                let inv = 1.0 / sum;
                for s in scores.iter_mut() {
                    *s *= inv;
                }
            }
            // out_head[qi, d] = Σ_ki scores[ki] * v[h, ki, d]  (sequential over ki)
            let o = &mut head_out[dst_base + qi * head_dim..dst_base + (qi + 1) * head_dim];
            for d in o.iter_mut() {
                *d = 0.0;
            }
            for (ki, &w) in scores.iter().enumerate() {
                let v_row = &v[head_off + ki * head_dim..head_off + (ki + 1) * head_dim];
                for d in 0..head_dim {
                    o[d] += w * v_row[d];
                }
            }
        }
    }

    // Transpose [num_heads, seq, head_dim] → [seq, num_heads*head_dim].
    let mut out = vec![0.0f32; seq * inner];
    for h in 0..num_heads {
        for qi in 0..seq {
            let src = &head_out[(h * seq + qi) * head_dim..(h * seq + qi + 1) * head_dim];
            let dst = &mut out[qi * inner + h * head_dim..qi * inner + (h + 1) * head_dim];
            dst.copy_from_slice(src);
        }
    }
    out
}

/// Deterministic pseudo-random fill in roughly `[-1, 1]` from an integer seed.
fn fill_deterministic(buf: &mut [f32], seed_base: u64) {
    // Simple LCG → mapped to [-1, 1]. Stable across platforms.
    let mut state = seed_base.wrapping_mul(6364136223846793005).wrapping_add(1);
    for x in buf.iter_mut() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (state >> 33) as u32;
        let unit = (bits as f32) / (u32::MAX as f32); // [0, 1]
        *x = unit * 2.0 - 1.0; // [-1, 1]
    }
}

/// Compute (max-abs error, cosine similarity) between two equal-length slices.
fn max_abs_and_cosine(a: &[f32], b: &[f32]) -> (f32, f64) {
    let mut max_abs = 0.0f32;
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        max_abs = max_abs.max((x - y).abs());
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    let cos = if na > 0.0 && nb > 0.0 {
        dot / (na.sqrt() * nb.sqrt())
    } else {
        1.0
    };
    (max_abs, cos)
}
/// Parity: the Metal flash-attention `joint_attention_flash_f32` kernel
/// (online-softmax + `simdgroup_float8x8` HW matrix units) must match the CPU
/// reference across the DiT shape and small / non-tile-multiple shapes — for the
/// standalone, pooled, and resident-dispatch entry points.
///
/// The online-softmax is mathematically exact vs the CPU full-row softmax (just
/// reordered), so f32 reassociation is the only source of error: gate is
/// max-abs < 1e-3 (expect ≈1e-5..1e-6) and cosine > 0.9999 on each shape/path.
#[test]
fn test_joint_attention_flash_parity() {
    // Serialize against the pool-alloc-count test: this test grows / uses the
    // process-wide joint-attention pool, which would otherwise perturb that
    // test's global alloc-count invariant when run in parallel.
    let _serial = JOINT_ATTN_POOL_TEST_SERIAL
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if Device::system_default().is_none() {
        eprintln!("test_joint_attention_flash_parity: no Metal device, skipping");
        return;
    }
    let graph = MetalGraph::new().expect("failed to create MetalGraph");

    // (num_heads, seq, head_dim): small, small, non-tile-multiple seq (50 is
    // neither a multiple of FA_BQ=64 nor FA_BK=32), a one-tile boundary (64),
    // and the DiT shape.
    let shapes: [(usize, usize, usize); 5] = [
        (2, 8, 128),
        (2, 40, 128),
        (1, 50, 128),
        (1, 64, 128),
        (24, 1536, 128),
    ];

    for (hi, &(num_heads, seq, head_dim)) in shapes.iter().enumerate() {
        let qkv_len = num_heads * seq * head_dim;
        let out_len = seq * num_heads * head_dim;

        let mut q = vec![0.0f32; qkv_len];
        let mut k = vec![0.0f32; qkv_len];
        let mut v = vec![0.0f32; qkv_len];
        fill_deterministic(&mut q, 0x4000 + hi as u64);
        fill_deterministic(&mut k, 0x5000 + hi as u64);
        fill_deterministic(&mut v, 0x6000 + hi as u64);

        let cpu_out = cpu_joint_attention_ref(&q, &k, &v, num_heads, seq, head_dim);

        // ── Path 1: standalone fresh-buffer flash encode ────────────────────
        let mut gpu_out = vec![0.0f32; out_len];
        graph
            .encode_joint_attention_flash(&q, &k, &v, &mut gpu_out, num_heads, seq, head_dim)
            .expect("encode_joint_attention_flash failed");
        let (max_abs, cos) = max_abs_and_cosine(&cpu_out, &gpu_out);
        eprintln!(
            "flash parity [standalone] (heads={num_heads}, seq={seq}, dim={head_dim}): \
             max_abs={max_abs:.3e}, cos={cos:.9}"
        );
        assert!(
            max_abs < 1e-3,
            "flash [standalone] parity FAIL (heads={num_heads}, seq={seq}, dim={head_dim}): \
             max_abs={max_abs:.3e} >= 1e-3 (cos={cos:.9})"
        );
        assert!(
            cos > 0.9999,
            "flash [standalone] cosine too low (heads={num_heads}, seq={seq}, dim={head_dim}): \
             cos={cos:.9}"
        );

        // ── Path 2: pooled (resident-buffer) flash encode ───────────────────
        let mut gpu_out_pool = vec![0.0f32; out_len];
        graph
            .encode_joint_attention_flash_pooled(
                &q,
                &k,
                &v,
                &mut gpu_out_pool,
                num_heads,
                seq,
                head_dim,
            )
            .expect("encode_joint_attention_flash_pooled failed");
        let (max_abs_p, cos_p) = max_abs_and_cosine(&cpu_out, &gpu_out_pool);
        eprintln!(
            "flash parity [pooled]     (heads={num_heads}, seq={seq}, dim={head_dim}): \
             max_abs={max_abs_p:.3e}, cos={cos_p:.9}"
        );
        assert!(
            max_abs_p < 1e-3,
            "flash [pooled] parity FAIL (heads={num_heads}, seq={seq}, dim={head_dim}): \
             max_abs={max_abs_p:.3e} >= 1e-3 (cos={cos_p:.9})"
        );
        // Same kernel/inputs as the standalone path → bit-identical.
        assert_eq!(
            gpu_out, gpu_out_pool,
            "flash pooled output differs from standalone (heads={num_heads}, seq={seq})"
        );

        // ── Path 3: resident prepare + flash dispatch + download ────────────
        graph
            .joint_attn_resident_prepare(&q, &k, &v, &gpu_out, num_heads, seq, head_dim)
            .expect("joint_attn_resident_prepare failed");
        graph
            .joint_attn_flash_resident_dispatch(num_heads, seq, head_dim)
            .expect("joint_attn_flash_resident_dispatch failed");
        let mut gpu_out_res = vec![0.0f32; out_len];
        graph
            .joint_attn_resident_download(&mut gpu_out_res)
            .expect("joint_attn_resident_download failed");
        let (max_abs_r, cos_r) = max_abs_and_cosine(&cpu_out, &gpu_out_res);
        eprintln!(
            "flash parity [resident]   (heads={num_heads}, seq={seq}, dim={head_dim}): \
             max_abs={max_abs_r:.3e}, cos={cos_r:.9}"
        );
        assert!(
            max_abs_r < 1e-3,
            "flash [resident] parity FAIL (heads={num_heads}, seq={seq}, dim={head_dim}): \
             max_abs={max_abs_r:.3e} >= 1e-3 (cos={cos_r:.9})"
        );
        assert_eq!(
            gpu_out, gpu_out_res,
            "flash resident output differs from standalone (heads={num_heads}, seq={seq})"
        );
    }
}

/// The pooled / resident path should engage its buffer pool: after the first
/// (warming) call the per-call allocation count stays flat across subsequent
/// same-or-smaller-shape calls — a deterministic, load-independent proof that
/// the buffers are reused (the resident-scenario premise of the benchmark).
#[test]
fn test_joint_attn_pool_alloc_count_is_constant_after_warmup() {
    // Serialize against the other pool-touching tests (parity/flash) so their
    // concurrent pool-grows do not bump the global counter inside this test's
    // before/after measurement window.
    let _serial = JOINT_ATTN_POOL_TEST_SERIAL
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if Device::system_default().is_none() {
        eprintln!("test_joint_attn_pool_alloc_count_is_constant_after_warmup: no Metal, skipping");
        return;
    }
    let graph = MetalGraph::new().expect("failed to create MetalGraph");
    let (num_heads, seq, head_dim) = (24usize, 1536usize, 128usize);
    let qkv_len = num_heads * seq * head_dim;
    let out_len = seq * num_heads * head_dim;
    let mut q = vec![0.0f32; qkv_len];
    let mut k = vec![0.0f32; qkv_len];
    let mut v = vec![0.0f32; qkv_len];
    fill_deterministic(&mut q, 0x51);
    fill_deterministic(&mut k, 0x62);
    fill_deterministic(&mut v, 0x73);
    let mut out = vec![0.0f32; out_len];

    // Warm the pool to the DiT shape.
    graph
        .encode_joint_attention_flash_pooled(&q, &k, &v, &mut out, num_heads, seq, head_dim)
        .expect("warm pooled call failed");

    let before = MetalGraph::joint_attn_pool_alloc_count();
    for _ in 0..8 {
        graph
            .encode_joint_attention_flash_pooled(&q, &k, &v, &mut out, num_heads, seq, head_dim)
            .expect("steady-state pooled call failed");
    }
    let after = MetalGraph::joint_attn_pool_alloc_count();
    assert_eq!(
        before,
        after,
        "joint-attn pool allocated {} more buffers across 8 steady-state calls (expected 0)",
        after - before
    );
}

/// Kernel-only (resident, q/k/v pre-uploaded, output left on GPU) **speed**
/// measurement of the flash-attention `joint_attention_flash_f32` kernel (the
/// shipping DiT attention path) at the DiT shape `(24, 1536, 128)`.
///
/// This is the GPU half of the honest benchmark: it reports the flash kernel's
/// kernel-only median ms + GFLOP/s. The CPU-real anchor (rayon+NEON
/// `oxibonsai_image::math::joint_attention`, ~305 ms) is measured by the
/// untouched `oxibonsai-image` bench (`tests/dit_attention_bench.rs`) on the same
/// machine — the kernels crate cannot depend on `oxibonsai-image`. `#[ignore]`
/// (run explicitly); informational only (no perf assertion — the parity tests
/// are the hard gate).
///
/// Run:
/// ```text
/// rustup run nightly cargo test -p oxibonsai-kernels --features metal --release \
///     bench_joint_attention_flash_kernel_only -- --ignored --nocapture
/// ```
#[test]
#[ignore = "GPU benchmark — run explicitly with --ignored --nocapture"]
fn bench_joint_attention_flash_kernel_only() {
    use std::time::Instant;

    if Device::system_default().is_none() {
        eprintln!("bench_joint_attention_flash_kernel_only: no Metal device, skipping");
        return;
    }
    let graph = MetalGraph::new().expect("failed to create MetalGraph");

    let (num_heads, seq, head_dim) = (24usize, 1536usize, 128usize);
    let qkv_len = num_heads * seq * head_dim;
    let out_len = seq * num_heads * head_dim;

    let mut q = vec![0.0f32; qkv_len];
    let mut k = vec![0.0f32; qkv_len];
    let mut v = vec![0.0f32; qkv_len];
    fill_deterministic(&mut q, 0xA1);
    fill_deterministic(&mut k, 0xB2);
    fill_deterministic(&mut v, 0xC3);

    // Parity guard: flash must match the CPU reference before we time it —
    // don't benchmark a broken result.
    let cpu_out = cpu_joint_attention_ref(&q, &k, &v, num_heads, seq, head_dim);
    let mut flash_out = vec![0.0f32; out_len];
    graph
        .encode_joint_attention_flash_pooled(&q, &k, &v, &mut flash_out, num_heads, seq, head_dim)
        .expect("flash pooled failed");
    let (parity, cos) = max_abs_and_cosine(&cpu_out, &flash_out);
    assert!(
        parity < 1e-3,
        "flash vs CPU max-abs {parity:e} exceeds 1e-3 — refusing to benchmark"
    );

    // Upload q/k/v into the resident pool ONCE; the kernel dispatches against it.
    graph
        .joint_attn_resident_prepare(&q, &k, &v, &cpu_out, num_heads, seq, head_dim)
        .expect("resident prepare failed");

    const WARMUP: usize = 5;
    const ITERS: usize = 15; // odd, >= 11

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
        times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        times[times.len() / 2]
    }

    // Two reps so the kernel sees comparable machine state across the window.
    let flash_a = measure(WARMUP, ITERS, || {
        graph
            .joint_attn_flash_resident_dispatch(num_heads, seq, head_dim)
            .expect("flash resident dispatch failed");
    });
    let flash_b = measure(WARMUP, ITERS, || {
        graph
            .joint_attn_flash_resident_dispatch(num_heads, seq, head_dim)
            .expect("flash resident dispatch failed");
    });

    let flash_ms = (flash_a + flash_b) / 2.0;

    // Attention FLOPs ≈ 2·H·N·N·D·2 (Q·Kᵀ and P·V, each 2·N·N·D per head).
    let gflop =
        2.0 * (num_heads as f64) * (seq as f64) * (seq as f64) * (head_dim as f64) * 2.0 / 1e9;
    let flash_gflops = gflop / (flash_ms / 1e3);

    let load = std::process::Command::new("uptime")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default();

    eprintln!("\n══════════════════════════════════════════════════════════════════════");
    eprintln!(
        "DiT joint-attention KERNEL-ONLY (heads={num_heads}, seq={seq}, head_dim={head_dim}); \
         flash-vs-CPU parity max-abs={parity:e}, cos={cos:.6}; \
         median of {ITERS}, warmup {WARMUP}, 2 reps; ≈{gflop:.1} GFLOP"
    );
    eprintln!(
        "  flash joint_attention_flash_f32  = {flash_ms:8.3} ms   [{flash_gflops:7.1} GFLOP/s]"
    );
    eprintln!("  (CPU real rayon+NEON anchor ≈ 305 ms — see oxibonsai-image dit_attention_bench)");
    if !load.is_empty() {
        eprintln!("  loadavg: {load}");
    }
    eprintln!("══════════════════════════════════════════════════════════════════════\n");
}
