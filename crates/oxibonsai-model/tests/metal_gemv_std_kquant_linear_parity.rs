//! Model-layer Metal GEMV short-circuit parity tests.
//!
//! Prior to this wave, `LinearQ4_0`/`LinearQ8_0`/`LinearQ2K`/`LinearQ3K`/
//! `LinearQ4K`/`LinearQ5K`/`LinearQ6K`/`LinearQ8K::forward` never reached the
//! parity-validated `oxibonsai_kernels::metal_gemv_*` kernels: the CUDA branch
//! existed, but there was no `metal` counterpart, so every call on macOS ran
//! the CPU scalar path even with `--features metal` enabled and a GPU
//! present.
//!
//! These tests build a small `Linear*` layer over synthetic quantized blocks
//! and confirm `forward()` — which now tries the Metal kernel first — agrees
//! with the CPU scalar free function (`oxibonsai_kernels::gemv_*`) within the
//! documented parity tolerance. `forward()` itself falls back to the CPU
//! scalar path internally on any Metal failure (including "no device"), so
//! these tests pass unconditionally on both GPU-equipped hosts (exercising
//! the new Metal short-circuit) and CI runners without one (exercising the
//! fallback) — either way, the assertion is that the two paths agree.

#![cfg(all(feature = "metal", target_os = "macos"))]

use oxibonsai_core::{
    BlockQ2K, BlockQ3K, BlockQ4K, BlockQ4_0, BlockQ5K, BlockQ6K, BlockQ8K, BlockQ8_0,
};
use oxibonsai_model::layers::linear_kquant_ext::{LinearQ5K, LinearQ6K};
use oxibonsai_model::layers::linear_kquant_full::{LinearQ2K, LinearQ3K, LinearQ4K, LinearQ8K};
use oxibonsai_model::layers::linear_standard::{LinearQ4_0, LinearQ8_0};

/// Deterministic, mixed-sign, positive-biased weight matrix (row-major).
fn weights(n_rows: usize, in_features: usize) -> Vec<f32> {
    (0..n_rows * in_features)
        .map(|idx| {
            let r = (idx / in_features) as f32;
            let c = (idx % in_features) as f32;
            (r * 0.13 + c * 0.07).sin() * 2.0 + (c * 0.031).cos() * 1.0 + 1.2
        })
        .collect()
}

/// Deterministic all-positive input vector of length `in_features`.
fn input(in_features: usize) -> Vec<f32> {
    (0..in_features)
        .map(|i| 0.5 + 0.4 * ((i as f32) * 0.05).sin())
        .collect()
}

fn assert_parity(label: &str, n_rows: usize, cpu: &[f32], gpu: &[f32]) {
    for row in 0..n_rows {
        let diff = (cpu[row] - gpu[row]).abs();
        let rel = diff / cpu[row].abs().max(1e-6);
        assert!(
            diff < 5e-3 || rel < 1e-3,
            "{label} parity failed: row={row} cpu={} gpu={} diff={diff} rel={rel}",
            cpu[row],
            gpu[row]
        );
    }
}

#[test]
fn linear_q4_0_forward_uses_metal_short_circuit_with_parity() {
    let n_rows = 33; // exercises the 32-lane stride tail
    let in_features = 1056;
    let w = weights(n_rows, in_features);
    let blocks = BlockQ4_0::quantize(&w).expect("Q4_0 quantize");
    let inp = input(in_features);

    let layer = LinearQ4_0::new(&blocks, n_rows, in_features).expect("LinearQ4_0::new");
    let mut gpu_or_cpu = vec![0.0f32; n_rows];
    // On a host without a Metal device this still succeeds (falls back to
    // CPU internally), so there is nothing to skip here — the point of this
    // test is only meaningful (exercises the Metal path) when a device is
    // present, but it must not fail either way.
    layer
        .forward(&inp, &mut gpu_or_cpu)
        .expect("LinearQ4_0::forward should succeed (metal or CPU fallback)");

    let mut cpu = vec![0.0f32; n_rows];
    oxibonsai_kernels::gemv_q4_0(&blocks, &inp, &mut cpu, n_rows, in_features)
        .expect("scalar Q4_0 GEMV");

    assert_parity("Q4_0", n_rows, &cpu, &gpu_or_cpu);
}

#[test]
fn linear_q8_0_forward_uses_metal_short_circuit_with_parity() {
    let n_rows = 33;
    let in_features = 1056;
    let w = weights(n_rows, in_features);
    let blocks = BlockQ8_0::quantize(&w).expect("Q8_0 quantize");
    let inp = input(in_features);

    let layer = LinearQ8_0::new(&blocks, n_rows, in_features).expect("LinearQ8_0::new");
    let mut gpu_or_cpu = vec![0.0f32; n_rows];
    layer
        .forward(&inp, &mut gpu_or_cpu)
        .expect("LinearQ8_0::forward should succeed (metal or CPU fallback)");

    let mut cpu = vec![0.0f32; n_rows];
    oxibonsai_kernels::gemv_q8_0(&blocks, &inp, &mut cpu, n_rows, in_features)
        .expect("scalar Q8_0 GEMV");

    assert_parity("Q8_0", n_rows, &cpu, &gpu_or_cpu);
}

#[test]
fn linear_q2k_forward_uses_metal_short_circuit_with_parity() {
    let n_rows = 5;
    let in_features = 512; // 2 super-blocks of 256
    let w = weights(n_rows, in_features);
    let blocks = BlockQ2K::quantize(&w).expect("Q2_K quantize");
    let inp = input(in_features);

    let layer = LinearQ2K::new(&blocks, n_rows, in_features).expect("LinearQ2K::new");
    let mut gpu_or_cpu = vec![0.0f32; n_rows];
    layer
        .forward(&inp, &mut gpu_or_cpu)
        .expect("LinearQ2K::forward should succeed (metal or CPU fallback)");

    let mut cpu = vec![0.0f32; n_rows];
    oxibonsai_kernels::gemv_q2k(&blocks, &inp, &mut cpu, n_rows, in_features)
        .expect("scalar Q2_K GEMV");

    assert_parity("Q2_K", n_rows, &cpu, &gpu_or_cpu);
}

#[test]
fn linear_q3k_forward_uses_metal_short_circuit_with_parity() {
    let n_rows = 5;
    let in_features = 512;
    let w = weights(n_rows, in_features);
    let blocks = BlockQ3K::quantize(&w).expect("Q3_K quantize");
    let inp = input(in_features);

    let layer = LinearQ3K::new(&blocks, n_rows, in_features).expect("LinearQ3K::new");
    let mut gpu_or_cpu = vec![0.0f32; n_rows];
    layer
        .forward(&inp, &mut gpu_or_cpu)
        .expect("LinearQ3K::forward should succeed (metal or CPU fallback)");

    let mut cpu = vec![0.0f32; n_rows];
    oxibonsai_kernels::gemv_q3k(&blocks, &inp, &mut cpu, n_rows, in_features)
        .expect("scalar Q3_K GEMV");

    assert_parity("Q3_K", n_rows, &cpu, &gpu_or_cpu);
}

#[test]
fn linear_q4k_forward_uses_metal_short_circuit_with_parity() {
    let n_rows = 5;
    let in_features = 512;
    let w = weights(n_rows, in_features);
    let blocks = BlockQ4K::quantize(&w).expect("Q4_K quantize");
    let inp = input(in_features);

    let layer = LinearQ4K::new(&blocks, n_rows, in_features).expect("LinearQ4K::new");
    let mut gpu_or_cpu = vec![0.0f32; n_rows];
    layer
        .forward(&inp, &mut gpu_or_cpu)
        .expect("LinearQ4K::forward should succeed (metal or CPU fallback)");

    let mut cpu = vec![0.0f32; n_rows];
    oxibonsai_kernels::gemv_q4k(&blocks, &inp, &mut cpu, n_rows, in_features)
        .expect("scalar Q4_K GEMV");

    assert_parity("Q4_K", n_rows, &cpu, &gpu_or_cpu);
}

#[test]
fn linear_q5k_forward_uses_metal_short_circuit_with_parity() {
    let n_rows = 5;
    let in_features = 512;
    let w = weights(n_rows, in_features);
    let blocks = BlockQ5K::quantize(&w).expect("Q5_K quantize");
    let inp = input(in_features);

    let layer = LinearQ5K::new(&blocks, n_rows, in_features).expect("LinearQ5K::new");
    let mut gpu_or_cpu = vec![0.0f32; n_rows];
    layer
        .forward(&inp, &mut gpu_or_cpu)
        .expect("LinearQ5K::forward should succeed (metal or CPU fallback)");

    let mut cpu = vec![0.0f32; n_rows];
    oxibonsai_kernels::gemv_q5k(&blocks, &inp, &mut cpu, n_rows, in_features)
        .expect("scalar Q5_K GEMV");

    assert_parity("Q5_K", n_rows, &cpu, &gpu_or_cpu);
}

#[test]
fn linear_q6k_forward_uses_metal_short_circuit_with_parity() {
    let n_rows = 5;
    let in_features = 512;
    let w = weights(n_rows, in_features);
    let blocks = BlockQ6K::quantize(&w).expect("Q6_K quantize");
    let inp = input(in_features);

    let layer = LinearQ6K::new(&blocks, n_rows, in_features).expect("LinearQ6K::new");
    let mut gpu_or_cpu = vec![0.0f32; n_rows];
    layer
        .forward(&inp, &mut gpu_or_cpu)
        .expect("LinearQ6K::forward should succeed (metal or CPU fallback)");

    let mut cpu = vec![0.0f32; n_rows];
    oxibonsai_kernels::gemv_q6k(&blocks, &inp, &mut cpu, n_rows, in_features)
        .expect("scalar Q6_K GEMV");

    assert_parity("Q6_K", n_rows, &cpu, &gpu_or_cpu);
}

#[test]
fn linear_q8k_forward_uses_metal_short_circuit_with_parity() {
    let n_rows = 5;
    let in_features = 512;
    let w = weights(n_rows, in_features);
    let blocks = BlockQ8K::quantize(&w).expect("Q8_K quantize");
    let inp = input(in_features);

    let layer = LinearQ8K::new(&blocks, n_rows, in_features).expect("LinearQ8K::new");
    let mut gpu_or_cpu = vec![0.0f32; n_rows];
    layer
        .forward(&inp, &mut gpu_or_cpu)
        .expect("LinearQ8K::forward should succeed (metal or CPU fallback)");

    let mut cpu = vec![0.0f32; n_rows];
    oxibonsai_kernels::gemv_q8k(&blocks, &inp, &mut cpu, n_rows, in_features)
        .expect("scalar Q8_K GEMV");

    assert_parity("Q8_K", n_rows, &cpu, &gpu_or_cpu);
}
