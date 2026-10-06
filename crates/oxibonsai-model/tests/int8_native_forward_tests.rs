//! Full-forward evidence for the opt-in INT8 tier (K-14) on the native
//! weight formats: the real `Ternary-Bonsai-1.7B` (`TQ2_0_g128`) and
//! `Bonsai-8B` (`Q1_0_g128`) driven end to end through [`BonsaiModel`] on
//! the best CPU tier, once with `OXIBONSAI_KERNEL_TIER` unset (the f32
//! default) and once with it set to `neon-dot` (clamped to what this CPU
//! supports), exactly as a user opts in.
//!
//! ## What is asserted
//!
//! For each of three prompts: the f32 run prefills and then decodes
//! [`STEPS`] greedy tokens; the INT8 run prefills the same prompt and is
//! **teacher-forced** with the f32 run's tokens, so both runs see the same
//! inputs at every position. At every one of those `1 + STEPS` logit
//! vectors (the prefill's last position, then each decode step):
//!
//! - `cos(f32 logits, int8 logits) >= 0.999`, and every logit is finite;
//! - the INT8 run differs from the f32 run somewhere — proof that the
//!   environment variable reached the model's GEMV/GEMM path at all.
//!
//! Argmax agreement per step is reported (a near-tie may legitimately flip
//! under int8 activation quantization, so it is evidence, not a gate), as
//! is the decode throughput of both runs and the machine's load.
//!
//! ## Why the model is built inside a `CpuOnlyBackendScope`
//!
//! `BonsaiModel::from_gguf` builds its layers' dispatchers with
//! `KernelDispatcher::auto_detect()`, which on an `--all-features` Mac is
//! `KernelTier::Gpu` — a tier the INT8 selector deliberately never diverts.
//! A model built that way would show `cos = 1.0` and prove nothing. The
//! scope makes those internal dispatchers land on the CPU tier, which is
//! exactly what the engine's `Backend::Cpu` does.
//!
//! ## Running
//!
//! The models are located only through `OXIBONSAI_MODELS_DIR` (else the
//! workspace's own `models/`); a missing model self-skips with a
//! `legacy-models` capability record (`executed: false`) and is never
//! `#[ignore]`d. Run in release with `--test-threads=1`: each test keeps one
//! real model resident.

use std::time::{Duration, Instant};

use oxibonsai_core::gguf::reader::{mmap_gguf_file, GgufFile};
use oxibonsai_kernels::dispatch::{cpu_kernel_tier, KernelDispatcher};
use oxibonsai_kernels::dispatch_int8::{Int8Tier, KERNEL_TIER_ENV};
use oxibonsai_kernels::gpu_backend::CpuOnlyBackendScope;
use oxibonsai_model::model::BonsaiModel;
use oxibonsai_testkit::capability::{record_executed_timed, record_skipped, Capability};

/// Teacher-forced decode steps per prompt, after the prefill.
const STEPS: usize = 8;

/// Context budget: the longest prompt plus [`STEPS`], with headroom.
const MAX_SEQ: usize = 128;

/// The accuracy gate.
const COS_GATE: f32 = 0.999;

/// The INT8 tier the opt-in runs ask for.
const REQUESTED_TIER: &str = "neon-dot";

/// Deterministic prompt token-id sequences — three lengths and patterns,
/// numeric rather than tokenized text because this crate has no tokenizer
/// dependency (the same convention `legacy_parity_tests.rs` uses).
fn prompt(slot: usize) -> Vec<u32> {
    match slot {
        0 => (0..24u32).map(|i| 1000 + i * 37).collect(),
        1 => (0..18u32).map(|i| 2000 + i * 53).collect(),
        _ => (0..30u32).map(|i| 500 + i * 19).collect(),
    }
}

/// Serializes every test in this binary against [`KERNEL_TIER_ENV`]:
/// `std::env::set_var` is `unsafe` because a concurrent read on any key can
/// observe a torn `environ`, and the tests run as threads of one process.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// RAII owner of [`KERNEL_TIER_ENV`] for one test: takes [`ENV_LOCK`],
/// snapshots and clears the variable (so an ambient shell export cannot turn
/// the "f32 default" run into an INT8 one), and restores the snapshot on
/// drop — also while unwinding from a failed assertion.
struct TierEnvGuard {
    _lock: std::sync::MutexGuard<'static, ()>,
    prior: Option<String>,
}

impl TierEnvGuard {
    fn acquire() -> Self {
        let lock = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let prior = std::env::var(KERNEL_TIER_ENV).ok();
        // SAFETY: `lock` is held for the lifetime of the returned guard, and
        // it serializes every reader and writer of the variable in this
        // binary.
        unsafe {
            std::env::remove_var(KERNEL_TIER_ENV);
        }
        Self { _lock: lock, prior }
    }

    /// Set (`Some`) or clear (`None`) the tier selector.
    fn select(&self, name: Option<&str>) {
        // SAFETY: `self._lock` is held (see `acquire`).
        unsafe {
            match name {
                Some(n) => std::env::set_var(KERNEL_TIER_ENV, n),
                None => std::env::remove_var(KERNEL_TIER_ENV),
            }
        }
    }
}

impl Drop for TierEnvGuard {
    fn drop(&mut self) {
        // SAFETY: `self._lock` is held for the entire body of `drop`.
        unsafe {
            match &self.prior {
                Some(v) => std::env::set_var(KERNEL_TIER_ENV, v),
                None => std::env::remove_var(KERNEL_TIER_ENV),
            }
        }
    }
}

/// First-index argmax (the engine's greedy tie-break).
fn argmax_first(values: &[f32]) -> u32 {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i as u32
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += f64::from(*x) * f64::from(*y);
        na += f64::from(*x) * f64::from(*x);
        nb += f64::from(*y) * f64::from(*y);
    }
    if na == 0.0 || nb == 0.0 {
        return if na == nb { 1.0 } else { 0.0 };
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

/// The 1/5/15-minute load average, for the throughput lines.
fn load_average() -> String {
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

/// One run over one prompt: `1 + STEPS` logit vectors, the greedy token at
/// each of them, and the wall-clock split.
struct Run {
    logits: Vec<Vec<f32>>,
    argmax: Vec<u32>,
    prefill: Duration,
    decode: Duration,
}

impl Run {
    fn decode_tok_s(&self) -> f64 {
        STEPS as f64 / self.decode.as_secs_f64().max(1e-12)
    }
}

/// Prefill `prompt`, then decode [`STEPS`] tokens: the run's own greedy
/// tokens when `forced` is `None`, else `forced[step]` at each step.
fn run(
    model: &mut BonsaiModel<'_>,
    kernel: &KernelDispatcher,
    prompt: &[u32],
    forced: Option<&[u32]>,
) -> Run {
    model.reset();
    let t = Instant::now();
    let first = model.forward_prefill(prompt, 0, kernel).expect("prefill");
    let prefill = t.elapsed();
    let mut argmax = vec![argmax_first(&first)];
    let mut logits = vec![first];
    let t = Instant::now();
    for step in 0..STEPS {
        let token = match forced {
            Some(tokens) => tokens[step],
            None => argmax[step],
        };
        let next = model
            .forward(token, prompt.len() + step, kernel)
            .expect("decode step");
        argmax.push(argmax_first(&next));
        logits.push(next);
    }
    Run {
        logits,
        argmax,
        prefill,
        decode: t.elapsed(),
    }
}

/// The whole gate for one real model: see the module doc.
fn check_model(file_name: &str, test_name: &str) {
    let guard = TierEnvGuard::acquire();
    let Some(path) = oxibonsai_testkit::workspace::find_model(file_name) else {
        eprintln!(
            "skip: {file_name} not present under {:?} (set OXIBONSAI_MODELS_DIR)",
            oxibonsai_testkit::workspace::models_dir()
        );
        record_skipped(Capability::LegacyModels, test_name);
        return;
    };
    let gate_start = std::time::Instant::now();
    let tier = Int8Tier::from_name(REQUESTED_TIER)
        .map(Int8Tier::clamp_to_cpu)
        .unwrap_or(Int8Tier::Scalar);

    let mmap = mmap_gguf_file(&path).expect("mmap the real GGUF");
    let gguf = GgufFile::parse(&mmap).expect("parse the real GGUF");
    let mut model = {
        let _cpu_only = CpuOnlyBackendScope::enter();
        BonsaiModel::from_gguf(&gguf, MAX_SEQ).expect("BonsaiModel::from_gguf")
    };
    let kernel = KernelDispatcher::with_tier(cpu_kernel_tier());
    assert_eq!(
        kernel.native_int8_tier(),
        None,
        "the f32 runs must start from a clean environment"
    );

    println!(
        "{file_name}: CPU tier {}, INT8 tier requested {REQUESTED_TIER} -> effective {tier}, \
         load average {}",
        kernel.tier(),
        load_average()
    );
    let (mut worst_cos, mut agree, mut total) = (1.0f32, 0usize, 0usize);
    let (mut f32_decode, mut int8_decode) = (Duration::ZERO, Duration::ZERO);
    for slot in 0..3 {
        let prompt = prompt(slot);

        guard.select(None);
        let base = run(&mut model, &kernel, &prompt, None);
        let forced: Vec<u32> = base.argmax[..STEPS].to_vec();

        guard.select(Some(tier.name()));
        assert_eq!(kernel.native_int8_tier(), Some(tier));
        let int8 = run(&mut model, &kernel, &prompt, Some(&forced));
        guard.select(None);

        let mut any_bits_differ = false;
        for (step, (a, b)) in base.logits.iter().zip(int8.logits.iter()).enumerate() {
            assert_eq!(a.len(), b.len(), "{file_name} slot {slot} step {step}");
            assert!(
                a.iter().chain(b.iter()).all(|v| v.is_finite()),
                "{file_name} slot {slot} step {step}: non-finite logits"
            );
            any_bits_differ |= a
                .iter()
                .zip(b.iter())
                .any(|(x, y)| x.to_bits() != y.to_bits());
            let cos = cosine(a, b);
            let (fa, ia) = (base.argmax[step], int8.argmax[step]);
            let what = if step == 0 { "prefill" } else { "decode " };
            println!(
                "{file_name} prompt {slot} {what} step {step}: cos = {cos:.6}, argmax f32 {fa} \
                 int8 {ia} {}",
                if fa == ia { "(agree)" } else { "(DIFFER)" }
            );
            worst_cos = worst_cos.min(cos);
            agree += usize::from(fa == ia);
            total += 1;
            assert!(
                cos >= COS_GATE,
                "{file_name} prompt {slot} step {step}: cos {cos} < {COS_GATE}"
            );
        }
        assert!(
            any_bits_differ,
            "{file_name} prompt {slot}: {KERNEL_TIER_ENV}={tier} produced the f32 logits bit \
             for bit — the tier never reached the forward path"
        );
        println!(
            "{file_name} prompt {slot}: decode {:.2} tok/s f32 -> {:.2} tok/s int8 ({tier}); \
             prefill of {} tokens {:?} f32 -> {:?} int8",
            base.decode_tok_s(),
            int8.decode_tok_s(),
            prompt.len(),
            base.prefill,
            int8.prefill,
        );
        f32_decode += base.decode;
        int8_decode += int8.decode;
    }
    let steps = (3 * STEPS) as f64;
    println!(
        "{file_name}: worst cos = {worst_cos:.6} over {total} teacher-forced logit vectors; \
         argmax agreement {agree}/{total}; decode {:.2} tok/s f32 -> {:.2} tok/s int8 \
         ({tier}), load average {}",
        steps / f32_decode.as_secs_f64().max(1e-12),
        steps / int8_decode.as_secs_f64().max(1e-12),
        load_average()
    );
    // Written only here, after every assertion above has passed.
    record_executed_timed(Capability::LegacyModels, test_name, gate_start.elapsed());
}

#[test]
fn int8_tier_full_forward_matches_f32_on_the_real_ternary_1_7b() {
    check_model(
        "Ternary-Bonsai-1.7B.gguf",
        "oxibonsai-model::int8_native_forward_tests::\
         int8_tier_full_forward_matches_f32_on_the_real_ternary_1_7b",
    );
}

#[test]
fn int8_tier_full_forward_matches_f32_on_the_real_bonsai_8b() {
    check_model(
        "Bonsai-8B.gguf",
        "oxibonsai-model::int8_native_forward_tests::\
         int8_tier_full_forward_matches_f32_on_the_real_bonsai_8b",
    );
}
