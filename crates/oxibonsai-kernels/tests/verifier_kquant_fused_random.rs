//! Adversarial cross-check of the
//! fused NEON K-quant GEMV kernels against an INDEPENDENT, from-scratch
//! transliteration of ggml's dequant formulas (written from ggml-quants.c,
//! not from `BlockQ*K::dequant` nor from the kernels under test).
//!
//! Uses RANDOM byte fills rather than `quantize()` round-trips, so the
//! Q4_K packed-scale high-bit branch of `get_scale_min_k4` (j >= 4, with
//! q[j-4] >> 6 != 0) is actually exercised — the in-file hand-derived
//! golden uses scales whose top two bits are all zero and therefore never
//! reaches that term.

use half::f16;
use oxibonsai_core::{BlockQ4K, BlockQ6K, BlockQ8K};

/// Deterministic xorshift, so failures are reproducible.
struct Rng(u64);
impl Rng {
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 32) as u32
    }
    fn byte(&mut self) -> u8 {
        (self.next_u32() & 0xFF) as u8
    }
    fn f32_pm(&mut self, scale: f32) -> f32 {
        ((self.next_u32() as f64 / u32::MAX as f64) as f32 - 0.5) * 2.0 * scale
    }
}

// ── independent references (ggml-quants.c transliterations) ──

fn ref_scale_min_k4(j: usize, q: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

fn ref_q4k_row(blocks: &[BlockQ4K]) -> Vec<f32> {
    let mut out = Vec::with_capacity(blocks.len() * 256);
    for b in blocks {
        let d = b.d.to_f32();
        let dmin = b.dmin.to_f32();
        for g in 0..4usize {
            let (sc1, m1) = ref_scale_min_k4(2 * g, &b.scales);
            let (sc2, m2) = ref_scale_min_k4(2 * g + 1, &b.scales);
            for l in 0..32usize {
                out.push(
                    d * f32::from(sc1) * f32::from(b.qs[g * 32 + l] & 0xF) - dmin * f32::from(m1),
                );
            }
            for l in 0..32usize {
                out.push(
                    d * f32::from(sc2) * f32::from(b.qs[g * 32 + l] >> 4) - dmin * f32::from(m2),
                );
            }
        }
    }
    out
}

fn ref_q6k_row(blocks: &[BlockQ6K]) -> Vec<f32> {
    let mut out = vec![0.0f32; blocks.len() * 256];
    for (bi, b) in blocks.iter().enumerate() {
        let d = b.d.to_f32();
        for n in 0..2usize {
            let (ql_off, qh_off, sc_off, y) = (n * 64, n * 32, n * 8, bi * 256 + n * 128);
            for l in 0..32usize {
                let is = l / 16;
                let qh = b.qh[qh_off + l];
                let q1 = i32::from((b.ql[ql_off + l] & 0xF) | ((qh & 3) << 4)) - 32;
                let q2 = i32::from((b.ql[ql_off + l + 32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32;
                let q3 = i32::from((b.ql[ql_off + l] >> 4) | (((qh >> 4) & 3) << 4)) - 32;
                let q4 = i32::from((b.ql[ql_off + l + 32] >> 4) | (((qh >> 6) & 3) << 4)) - 32;
                out[y + l] = d * f32::from(b.scales[sc_off + is]) * q1 as f32;
                out[y + l + 32] = d * f32::from(b.scales[sc_off + is + 2]) * q2 as f32;
                out[y + l + 64] = d * f32::from(b.scales[sc_off + is + 4]) * q3 as f32;
                out[y + l + 96] = d * f32::from(b.scales[sc_off + is + 6]) * q4 as f32;
            }
        }
    }
    out
}

fn ref_q8k_row(blocks: &[BlockQ8K]) -> Vec<f32> {
    let mut out = Vec::with_capacity(blocks.len() * 256);
    for b in blocks {
        for i in 0..256usize {
            out.push(b.d * f32::from(b.qs[i]));
        }
    }
    out
}

fn dot(w: &[f32], x: &[f32]) -> f64 {
    w.iter()
        .zip(x.iter())
        .map(|(a, b)| f64::from(*a) * f64::from(*b))
        .sum()
}

fn check(got: f32, expected: f64, tag: &str) {
    let tol = 1e-4 * expected.abs().max(1.0);
    assert!(
        (f64::from(got) - expected).abs() <= tol,
        "{tag}: got={got}, expected={expected}"
    );
}

#[test]
fn fused_q4k_matches_independent_ggml_reference_on_random_bytes() {
    let mut rng = Rng(0x2026_0921_1234_5678);
    for trial in 0..24usize {
        let bpr = 1 + trial % 3;
        let n_rows = 1 + trial % 4;
        let in_features = bpr * 256;
        let mut blocks = Vec::new();
        for _ in 0..n_rows * bpr {
            let mut qs = [0u8; 128];
            for q in qs.iter_mut() {
                *q = rng.byte();
            }
            let mut scales = [0u8; 12];
            for s in scales.iter_mut() {
                // Full 0..=255 range: exercises the >>6 high-bit term.
                *s = rng.byte();
            }
            blocks.push(BlockQ4K {
                d: f16::from_f32(rng.f32_pm(0.05)),
                dmin: f16::from_f32(rng.f32_pm(0.05)),
                scales,
                qs,
            });
        }
        let input: Vec<f32> = (0..in_features).map(|_| rng.f32_pm(2.0)).collect();
        let mut got = vec![0.0f32; n_rows];
        oxibonsai_kernels::gemv_q4k::gemv_q4k(&blocks, &input, &mut got, n_rows, in_features)
            .expect("gemv_q4k");
        for r in 0..n_rows {
            let w = ref_q4k_row(&blocks[r * bpr..(r + 1) * bpr]);
            check(
                got[r],
                dot(&w, &input),
                &format!("q4k trial={trial} row={r}"),
            );
        }
    }
}

#[test]
fn fused_q6k_matches_independent_ggml_reference_on_random_bytes() {
    let mut rng = Rng(0x0BAD_C0DE_DEAD_BEEF);
    for trial in 0..24usize {
        let bpr = 1 + trial % 3;
        let n_rows = 1 + trial % 4;
        let in_features = bpr * 256;
        let mut blocks = Vec::new();
        for _ in 0..n_rows * bpr {
            let mut ql = [0u8; 128];
            for q in ql.iter_mut() {
                *q = rng.byte();
            }
            let mut qh = [0u8; 64];
            for q in qh.iter_mut() {
                *q = rng.byte();
            }
            let mut scales = [0i8; 16];
            for s in scales.iter_mut() {
                *s = rng.byte() as i8;
            }
            blocks.push(BlockQ6K {
                ql,
                qh,
                scales,
                d: f16::from_f32(rng.f32_pm(0.02)),
            });
        }
        let input: Vec<f32> = (0..in_features).map(|_| rng.f32_pm(2.0)).collect();
        let mut got = vec![0.0f32; n_rows];
        oxibonsai_kernels::gemv_q6k::gemv_q6k(&blocks, &input, &mut got, n_rows, in_features)
            .expect("gemv_q6k");
        for r in 0..n_rows {
            let w = ref_q6k_row(&blocks[r * bpr..(r + 1) * bpr]);
            check(
                got[r],
                dot(&w, &input),
                &format!("q6k trial={trial} row={r}"),
            );
        }
    }
}

#[test]
fn fused_q8k_matches_independent_ggml_reference_on_random_bytes() {
    let mut rng = Rng(0x1357_9BDF_2468_ACE0);
    for trial in 0..24usize {
        let bpr = 1 + trial % 3;
        let n_rows = 1 + trial % 4;
        let in_features = bpr * 256;
        let mut blocks = Vec::new();
        for _ in 0..n_rows * bpr {
            let mut qs = [0i8; 256];
            for q in qs.iter_mut() {
                *q = rng.byte() as i8;
            }
            blocks.push(BlockQ8K {
                d: rng.f32_pm(0.01),
                qs,
                bsums: [0i16; 16],
            });
        }
        let input: Vec<f32> = (0..in_features).map(|_| rng.f32_pm(2.0)).collect();
        let mut got = vec![0.0f32; n_rows];
        oxibonsai_kernels::gemv_q8k::gemv_q8k(&blocks, &input, &mut got, n_rows, in_features)
            .expect("gemv_q8k");
        for r in 0..n_rows {
            let w = ref_q8k_row(&blocks[r * bpr..(r + 1) * bpr]);
            check(
                got[r],
                dot(&w, &input),
                &format!("q8k trial={trial} row={r}"),
            );
        }
    }
}

/// Large-row variant: forces the Rayon-parallel branch of the fused driver.
#[test]
fn fused_kernels_match_reference_in_the_parallel_branch() {
    let mut rng = Rng(0xFEED_FACE_CAFE_0001);
    let n_rows = 600usize;
    let bpr = 1usize;
    let in_features = bpr * 256;

    let mut q4 = Vec::new();
    for _ in 0..n_rows {
        let mut qs = [0u8; 128];
        for q in qs.iter_mut() {
            *q = rng.byte();
        }
        let mut scales = [0u8; 12];
        for s in scales.iter_mut() {
            *s = rng.byte();
        }
        q4.push(BlockQ4K {
            d: f16::from_f32(rng.f32_pm(0.05)),
            dmin: f16::from_f32(rng.f32_pm(0.05)),
            scales,
            qs,
        });
    }
    let input: Vec<f32> = (0..in_features).map(|_| rng.f32_pm(2.0)).collect();
    let mut got = vec![0.0f32; n_rows];
    oxibonsai_kernels::gemv_q4k::gemv_q4k(&q4, &input, &mut got, n_rows, in_features)
        .expect("gemv_q4k");
    for r in 0..n_rows {
        let w = ref_q4k_row(&q4[r..r + 1]);
        check(got[r], dot(&w, &input), &format!("q4k-par row={r}"));
    }
}
