//! Shared helpers of the INT8 tier acceptance suite: deterministic block
//! and activation generators, the bit-level and cosine comparisons every
//! accuracy assertion uses, the tiers this host can execute, and the
//! interleaved timing harness of the throughput reports.

use half::f16;
use oxibonsai_core::tensor::{BlockQ1_0G128, QK1_0_G128};
use oxibonsai_core::{BlockPQ2_0, BlockTQ2_0_g128};
use oxibonsai_kernels::dispatch_int8::Int8Tier;
use oxibonsai_kernels::KernelDispatcher;
use std::time::{Duration, Instant};

pub(super) struct Lcg(u32);

impl Lcg {
    pub(super) fn new(seed: u32) -> Self {
        Self(seed | 1)
    }
    pub(super) fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.0
    }
    pub(super) fn next_u8(&mut self) -> u8 {
        (self.next() >> 19) as u8
    }
    pub(super) fn next_f32(&mut self) -> f32 {
        ((self.next() >> 8) as i32 % 2001 - 1000) as f32 / 512.0
    }
}

pub(super) fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    if na == 0.0 || nb == 0.0 {
        return if na == nb { 1.0 } else { 0.0 };
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

pub(super) fn tq2_blocks(n: usize, seed: u32) -> Vec<BlockTQ2_0_g128> {
    let mut rng = Lcg::new(seed);
    (0..n)
        .map(|_| {
            let mut qs = [0u8; 32];
            for b in qs.iter_mut() {
                *b = rng.next_u8();
            }
            BlockTQ2_0_g128 {
                qs,
                d: f16::from_f32(0.0625 + (rng.next_u8() % 16) as f32 / 256.0),
            }
        })
        .collect()
}

pub(super) fn pq2_blocks(n: usize, seed: u32) -> Vec<BlockPQ2_0> {
    let mut rng = Lcg::new(seed);
    (0..n)
        .map(|_| {
            let mut qs = [0u8; 32];
            for b in qs.iter_mut() {
                *b = rng.next_u8();
            }
            BlockPQ2_0 {
                d: f16::from_f32(0.0625 + (rng.next_u8() % 16) as f32 / 256.0),
                qs,
            }
        })
        .collect()
}

pub(super) fn q1_blocks(n: usize, seed: u32) -> Vec<BlockQ1_0G128> {
    let mut rng = Lcg::new(seed);
    (0..n)
        .map(|_| {
            let mut qs = [0u8; QK1_0_G128 / 8];
            for b in qs.iter_mut() {
                *b = rng.next_u8();
            }
            BlockQ1_0G128 {
                d: f16::from_f32(0.125 + (rng.next_u8() % 16) as f32 / 256.0),
                qs,
            }
        })
        .collect()
}

pub(super) fn activations(len: usize, seed: u32) -> Vec<f32> {
    let mut rng = Lcg::new(seed);
    (0..len).map(|_| rng.next_f32()).collect()
}

/// Every INT8 tier this host can execute.
pub(super) fn supported_tiers() -> Vec<Int8Tier> {
    Int8Tier::ALL
        .iter()
        .copied()
        .filter(|t| t.is_supported())
        .collect()
}

/// A dispatcher pinned to this host's best **CPU** tier — never
/// `auto_detect`, which can land on `KernelTier::Gpu` on an
/// `--all-features` Mac, a tier the INT8 selector deliberately never
/// diverts (so an opt-in leg run on it would prove nothing).
pub(super) fn cpu_dispatcher() -> KernelDispatcher {
    KernelDispatcher::with_tier(oxibonsai_kernels::cpu_kernel_tier())
}

pub(super) fn assert_bits_eq(expect: &[f32], got: &[f32], what: &str) {
    assert_eq!(expect.len(), got.len(), "{what}: length");
    for (i, (e, g)) in expect.iter().zip(got.iter()).enumerate() {
        assert_eq!(
            e.to_bits(),
            g.to_bits(),
            "{what}: element {i} diverged: {e} vs {g}"
        );
    }
}

pub(super) fn bits_differ(a: &[f32], b: &[f32]) -> bool {
    a.iter()
        .zip(b.iter())
        .any(|(x, y)| x.to_bits() != y.to_bits())
}

/// Whether the opt-in wall-clock assertions are on.
pub(super) fn bench_asserts() -> bool {
    std::env::var("OXIBONSAI_INT8_BENCH").as_deref() == Ok("1")
}

/// The 1/5/15-minute load average, for the throughput lines (a ratio
/// measured at load 40 and one measured on an idle host are not the
/// same number). `"unknown"` where the host exposes neither source.
pub(super) fn load_average() -> String {
    if let Ok(s) = std::fs::read_to_string("/proc/loadavg") {
        return s.split_whitespace().take(3).collect::<Vec<_>>().join(" ");
    }
    std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| {
            s.trim()
                .trim_matches(|c| c == '{' || c == '}')
                .trim()
                .to_string()
        })
        .unwrap_or_else(|| "unknown".to_string())
}

/// An interleaved timing of two alternatives.
///
/// Each round times `baseline` and `candidate` back to back, the order
/// alternating between rounds, so both sides sample the same stretch of
/// machine load; a min-of-trials comparison of two loops timed one
/// after the other is decided by which loop happened to run in a quiet
/// minute on a shared build machine (load average 50-90 here).
///
/// The verdict, [`Paired::ratio`], is `baseline_min / candidate_min`:
/// interference from other processes only ever *adds* time, so each
/// side's minimum over the interleaved rounds is its least-disturbed
/// run. The median of the per-round ratios is kept for the report.
pub(super) struct Paired {
    pub(super) median_ratio: f64,
    pub(super) baseline_min: Duration,
    pub(super) candidate_min: Duration,
}

impl Paired {
    /// `baseline_min / candidate_min` (`> 1` means the candidate is
    /// faster).
    pub(super) fn ratio(&self) -> f64 {
        self.baseline_min.as_secs_f64() / self.candidate_min.as_secs_f64().max(1e-12)
    }
}

pub(super) fn paired(
    rounds: usize,
    mut baseline: impl FnMut(),
    mut candidate: impl FnMut(),
) -> Paired {
    let mut ratios = Vec::with_capacity(rounds.max(1));
    let mut baseline_min = Duration::MAX;
    let mut candidate_min = Duration::MAX;
    for round in 0..rounds.max(1) {
        let time = |f: &mut dyn FnMut()| {
            let t = Instant::now();
            f();
            t.elapsed()
        };
        let (b, c) = if round % 2 == 0 {
            let b = time(&mut baseline);
            (b, time(&mut candidate))
        } else {
            let c = time(&mut candidate);
            (time(&mut baseline), c)
        };
        baseline_min = baseline_min.min(b);
        candidate_min = candidate_min.min(c);
        ratios.push(b.as_secs_f64() / c.as_secs_f64().max(1e-12));
    }
    ratios.sort_by(f64::total_cmp);
    Paired {
        median_ratio: ratios[ratios.len() / 2],
        baseline_min,
        candidate_min,
    }
}
