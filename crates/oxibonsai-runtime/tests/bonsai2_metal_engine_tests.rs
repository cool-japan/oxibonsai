//! Real Bonsai 2 27B through the ordinary [`InferenceEngine`] on the Metal
//! hybrid runner — the product path `oxibonsai run`, `chat` and `serve`
//! take under `--backend auto` on a Metal host — for both release bands
//! (`PQ2_0`, `PTQ1_0`).
//!
//! * **G11-engine** (`real_27b_metal_engine_{pq2_0,ptq1_0}_matches_cpu_and_fork_bonsai2`)
//!   — for the three golden prompts, the Metal engine's 24 greedy ids equal
//!   the CPU engine's (`--backend cpu`), so the decoded texts are
//!   byte-identical, and they satisfy the G4 band against the fork's
//!   Metal golden at every step: greedy id exact, top-10 id set exact, worst
//!   `|Δlogprob|` ≤ 2e-2. The engine's own `generate` reproduces the ids.
//! * **G10-engine** (`real_27b_metal_engine_decode_throughput_and_memory_bonsai2`)
//!   — decode ≥ 5 tokens/s through the engine (best of three 32-token runs
//!   after a warm-up), a 512-token prefill's rate, the first-token latency,
//!   the memory, what keeping the CPU model beside the runner costs, and
//!   what a second runner in the same process adds. Memory is reported
//!   three ways: the resident set (which also counts mapped weight pages as
//!   the page cache lends them), the kernel's footprint ledger (dirty memory
//!   plus Metal allocations; clean file pages excluded) and
//!   `MTLDevice.currentAllocatedSize`, split into the runner's no-copy view
//!   of the mapped file — counted once per runner although every runner
//!   reads the same pages — and its own buffers.
//! * **server** (`real_27b_metal_engine_serves_chat_like_the_cpu_engine_bonsai2`)
//!   — one `/v1/chat/completions` round trip at temperature 0 answers with
//!   the same message on the Metal engine as on the CPU engine.
//!
//! Every test here holds [`TierEnvGuard`] for its whole run: it serialises
//! the binary (one 27B mapping and one decode at a time) and clears
//! `OXIBONSAI_KERNEL_TIER`, which the CPU model's `PQ2_0` GEMV would
//! otherwise honour (K-14) — flipping CPU-vs-Metal identity for a developer
//! who exported it. Two synthetic cases,
//! `metal_engine_int8_tier_log_names_the_case` and
//! `int8_tier_report_reads_every_case_from_the_environment`, set the
//! variable themselves to pin how each executor and format reports it; a
//! third, `metal_engine_decode_does_not_grow_the_process_footprint`, holds
//! the process footprint flat across 1550 Metal hybrid forwards on the
//! synthetic model.
//!
//! # Where the files come from
//!
//! Only from `OXI_BONSAI2_PQ2_GGUF` / `OXI_BONSAI2_PTQ1_GGUF` — no models
//! directory fallback, so a workspace test run never maps a 27B by accident
//! — and the fork's goldens from `OXI_BONSAI2_GOLDEN_DIR`, else the copy
//! vendored with the model crate's tests. A gate whose file is not set
//! skips with a `bonsai2-metal-engine` capability record
//! (`OXI_REQUIRE_MODEL_FILES=1` turns that into a failure); a gate that ran
//! records `executed`.

#[path = "../../oxibonsai-model/tests/bonsai2_real/harness.rs"]
mod harness;

use oxibonsai_kernels::dispatch_int8::KERNEL_TIER_ENV;

/// Capability-record name of G11-engine on `PQ2_0`.
const G11_PQ2_TEST: &str =
    "oxibonsai-runtime::bonsai2_metal_engine_tests::real_27b_metal_engine_pq2_0_matches_cpu_and_fork_bonsai2";
/// Capability-record name of G11-engine on `PTQ1_0`.
const G11_PTQ1_TEST: &str =
    "oxibonsai-runtime::bonsai2_metal_engine_tests::real_27b_metal_engine_ptq1_0_matches_cpu_and_fork_bonsai2";
/// Capability-record name of G10-engine.
const G10_TEST: &str =
    "oxibonsai-runtime::bonsai2_metal_engine_tests::real_27b_metal_engine_decode_throughput_and_memory_bonsai2";
/// Capability-record name of the server round trip.
const SERVER_TEST: &str =
    "oxibonsai-runtime::bonsai2_metal_engine_tests::real_27b_metal_engine_serves_chat_like_the_cpu_engine_bonsai2";

// ─────────────────────────────────────────────────────────────────────────
// The tier-variable guard (serialises the binary)
// ─────────────────────────────────────────────────────────────────────────

/// Serialises every test in this binary (see the module docs).
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// RAII owner of `OXIBONSAI_KERNEL_TIER` for one test: takes [`ENV_LOCK`],
/// snapshots and **clears** the variable, and restores the snapshot on drop
/// (also while unwinding from a failed assertion).
struct TierEnvGuard {
    _lock: std::sync::MutexGuard<'static, ()>,
    prior: Option<String>,
}

impl TierEnvGuard {
    fn cleared() -> Self {
        let lock = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let prior = std::env::var(KERNEL_TIER_ENV).ok();
        // SAFETY: `lock` is held for the lifetime of the returned guard and
        // serialises every reader and writer of the variable in this binary.
        unsafe {
            std::env::remove_var(KERNEL_TIER_ENV);
        }
        Self { _lock: lock, prior }
    }

    /// Set (`Some`) or clear (`None`) the tier selector.
    fn select(&self, name: Option<&str>) {
        // SAFETY: `self._lock` is held (see `cleared`).
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

// ─────────────────────────────────────────────────────────────────────────
// The gates
// ─────────────────────────────────────────────────────────────────────────

/// G11-engine on the real `PQ2_0` 27B.
#[test]
fn real_27b_metal_engine_pq2_0_matches_cpu_and_fork_bonsai2() {
    let _env = TierEnvGuard::cleared();
    #[cfg(all(feature = "metal", target_os = "macos"))]
    gates::g11_engine(&gates::PQ2_0, G11_PQ2_TEST);
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    skip_unbuilt(G11_PQ2_TEST, "the Metal backend is not compiled in");
}

/// G11-engine on the real `PTQ1_0` 27B.
#[test]
fn real_27b_metal_engine_ptq1_0_matches_cpu_and_fork_bonsai2() {
    let _env = TierEnvGuard::cleared();
    #[cfg(all(feature = "metal", target_os = "macos"))]
    gates::g11_engine(&gates::PTQ1_0, G11_PTQ1_TEST);
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    skip_unbuilt(G11_PTQ1_TEST, "the Metal backend is not compiled in");
}

/// G10-engine on both bands.
#[test]
fn real_27b_metal_engine_decode_throughput_and_memory_bonsai2() {
    let _env = TierEnvGuard::cleared();
    #[cfg(all(feature = "metal", target_os = "macos"))]
    gates::g10_engine(G10_TEST);
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    skip_unbuilt(G10_TEST, "the Metal backend is not compiled in");
}

/// The `/v1/chat/completions` round trip on both bands.
#[test]
fn real_27b_metal_engine_serves_chat_like_the_cpu_engine_bonsai2() {
    let _env = TierEnvGuard::cleared();
    #[cfg(all(feature = "metal", target_os = "macos", feature = "server"))]
    gates::server_round_trip(SERVER_TEST);
    #[cfg(not(all(feature = "metal", target_os = "macos", feature = "server")))]
    skip_unbuilt(
        SERVER_TEST,
        "the Metal backend or the server is not compiled in",
    );
}

/// How each executor reports `OXIBONSAI_KERNEL_TIER` (K-14) on the synthetic
/// `qwen35` fixture — unset, and set: the CPU engine's `PQ2_0` GEMV honours
/// it, the Metal engine's runner never does — and the Metal engine's logits
/// are bit-identical with the variable set and unset.
#[test]
fn metal_engine_int8_tier_log_names_the_case() {
    use oxibonsai_core::gguf::reader::GgufFile;
    use oxibonsai_runtime::engine::InferenceEngine;
    use oxibonsai_runtime::engine_seam::Backend;

    let env = TierEnvGuard::cleared();
    let bytes = oxibonsai_testkit::qwen35_fixture::synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let build = |backend| {
        InferenceEngine::from_gguf_with_backend(&gguf, greedy_params(8), 42, 64, backend)
            .expect("hybrid engine")
    };
    let prompt = [7u32, 11, 13, 17, 19];

    let cpu = build(Backend::Cpu);
    let unset = cpu.effective_tier_reason();
    assert!(unset.contains("OXIBONSAI_KERNEL_TIER unset"), "{unset}");

    env.select(Some("int8-scalar"));
    let honoured = cpu.effective_tier_reason();
    assert!(
        honoured.contains("OXIBONSAI_KERNEL_TIER=int8-scalar honoured: the PQ2_0 GEMV/GEMM"),
        "{honoured}"
    );
    env.select(None);

    if !metal_device_present() {
        // Only the CPU half applies without a Metal device (or build).
        return;
    }
    let mut metal = build(Backend::Metal);
    let unset_logits = metal.prefill_from_pos(&prompt, 0).expect("prefill");
    env.select(Some("int8-scalar"));
    let metal_reason = metal.effective_tier_reason();
    assert!(
        metal_reason.contains("Metal (hybrid runner)")
            && metal_reason.contains("honoured only by the CPU model's own PQ2_0 GEMV")
            && metal_reason.contains("never by the Metal hybrid runner"),
        "{metal_reason}"
    );
    metal.reset();
    let set_logits = metal.prefill_from_pos(&prompt, 0).expect("prefill");
    env.select(None);
    assert_eq!(
        bits(&unset_logits),
        bits(&set_logits),
        "the Metal runner never reads the INT8 tier selector"
    );
}

/// The four ways `OXIBONSAI_KERNEL_TIER` applies (K-14), read from this
/// process's environment exactly as the engine's tier log reads it: unset; a
/// native format on a CPU tier (honoured); a native format on the GPU tier
/// (set but ignored); a Prism format (honoured by the CPU model whatever its
/// tier, never by the Metal hybrid runner).
#[test]
fn int8_tier_report_reads_every_case_from_the_environment() {
    use oxibonsai_core::GgufTensorType as T;
    use oxibonsai_kernels::dispatch_int8::Int8Tier;
    use oxibonsai_runtime::engine_control::{int8_tier_use_from_env, Int8TierUse, TierExecutor};

    let env = TierEnvGuard::cleared();
    let cpu = oxibonsai_kernels::cpu_kernel_tier();
    assert_eq!(
        int8_tier_use_from_env(T::PQ2_0, cpu, TierExecutor::HybridMetal),
        Int8TierUse::Unset
    );

    env.select(Some("int8-scalar"));
    let tier = Int8Tier::Scalar;
    // (a) A native format on a CPU tier: honoured.
    assert_eq!(
        int8_tier_use_from_env(T::TQ2_0_g128, cpu, TierExecutor::Dense),
        Int8TierUse::Honoured {
            format: T::TQ2_0_g128,
            tier
        }
    );
    // (b) A native format on the GPU tier: set but ignored.
    #[cfg(feature = "metal")]
    assert_eq!(
        int8_tier_use_from_env(
            T::Q1_0_g128,
            oxibonsai_kernels::KernelTier::Gpu,
            TierExecutor::Dense
        ),
        Int8TierUse::IgnoredOnGpuTier {
            format: T::Q1_0_g128,
            tier
        }
    );
    // (c) A Prism format: honoured by the CPU model on any tier ...
    assert_eq!(
        int8_tier_use_from_env(T::PQ2_0, cpu, TierExecutor::HybridCpu),
        Int8TierUse::Honoured {
            format: T::PQ2_0,
            tier
        }
    );
    #[cfg(feature = "metal")]
    assert_eq!(
        int8_tier_use_from_env(
            T::PQ2_0,
            oxibonsai_kernels::KernelTier::Gpu,
            TierExecutor::HybridCpu
        ),
        Int8TierUse::HonouredDespiteGpuTier {
            format: T::PQ2_0,
            tier
        }
    );
    // ... and never by the Metal hybrid runner.
    assert_eq!(
        int8_tier_use_from_env(T::PQ2_0, cpu, TierExecutor::HybridMetal),
        Int8TierUse::CpuModelOnly {
            format: T::PQ2_0,
            tier
        }
    );

    env.select(None);
    assert_eq!(
        int8_tier_use_from_env(T::TQ2_0_g128, cpu, TierExecutor::Dense),
        Int8TierUse::Unset
    );
}

/// A Metal-backed hybrid engine's decode loop does not grow the process:
/// every forward's command buffer and encoder are autoreleased objects that
/// the runner drains per call (left to the thread they cost ~1.8 KiB per
/// decoded token until it exits).
#[test]
fn metal_engine_decode_does_not_grow_the_process_footprint() {
    let _env = TierEnvGuard::cleared();
    // Only a Metal build with a device has a runner to measure.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    if metal_device_present() {
        gates::decode_footprint_is_flat();
    }
}

/// A build without the Metal backend (or, for the round trip, the server)
/// cannot run these gates.
#[cfg(not(all(feature = "metal", target_os = "macos", feature = "server")))]
fn skip_unbuilt(test: &str, what: &str) {
    assert!(
        !harness::require_model_files(),
        "{}=1 but in this test binary {what}",
        harness::REQUIRE_ENV
    );
    eprintln!("skip {test}: {what}");
    record(false, test, None);
}

/// Record a gate's evidence under `bonsai2-metal-engine` — executed (with
/// the gate's wall time) or skipped — and print the record.
fn record(executed: bool, test: &str, duration: Option<std::time::Duration>) {
    use oxibonsai_testkit::capability::{
        record_executed, record_executed_timed, record_skipped, Capability,
    };
    let capability = Capability::Bonsai2MetalEngine;
    match (executed, duration) {
        (true, Some(d)) => record_executed_timed(capability, test, d),
        (true, None) => record_executed(capability, test),
        (false, _) => record_skipped(capability, test),
    }
    let suffix = duration.map_or_else(String::new, |d| format!(" duration_ms={}", d.as_millis()));
    eprintln!("CAPABILITY-REPORT capability={capability} executed={executed} test={test}{suffix}");
}

/// Whether this build and host have a Metal device (a missing device is the
/// only "no"; any other failure to open it is a test failure).
fn metal_device_present() -> bool {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        match oxibonsai_kernels::MetalGraph::shared_device() {
            Ok(_) => true,
            Err(oxibonsai_kernels::MetalGraphError::DeviceNotFound) => false,
            Err(e) => panic!("the Metal device must open on this host: {e}"),
        }
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        false
    }
}

fn greedy_params(max_tokens: usize) -> oxibonsai_runtime::sampling::SamplingParams {
    oxibonsai_runtime::sampling::SamplingParams {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens,
    }
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod gates {
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};

    use oxibonsai_core::gguf::reader::{mmap_gguf_file, GgufFile};
    use oxibonsai_runtime::engine::{tokenizer_from_gguf, InferenceEngine};
    use oxibonsai_runtime::engine_hybrid_gpu::HybridBackend;
    use oxibonsai_runtime::engine_seam::Backend;
    use oxibonsai_runtime::tokenizer_bridge::TokenizerBridge;

    use super::greedy_params;
    use super::harness::{
        argmax, compare_step, log_softmax, parse_golden_steps, parse_prompt_tokens,
        G4_LOGPROB_BAND, PQ2_ENV, PQ2_FILE, PROMPTS, PTQ1_ENV, PTQ1_FILE,
    };
    use super::record;

    /// A real-model file named by `env` alone, or `None` after recording the
    /// skip (a hard failure under `OXI_REQUIRE_MODEL_FILES=1`).
    fn locate(env: &str, file: &str, test: &str) -> Option<PathBuf> {
        let found = std::env::var(env)
            .ok()
            .filter(|v| !v.trim().is_empty())
            .map(PathBuf::from);
        match found {
            Some(path) => {
                assert!(
                    path.is_file(),
                    "{env} is set to {} but that is not a file",
                    path.display()
                );
                Some(path)
            }
            None => {
                assert!(
                    !super::harness::require_model_files(),
                    "{}=1 but {env} is not set (point it at {file})",
                    super::harness::REQUIRE_ENV
                );
                eprintln!("skip {test}: {env} is not set (point it at the real {file})");
                record(false, test, None);
                None
            }
        }
    }

    /// The fork's Metal goldens: `OXI_BONSAI2_GOLDEN_DIR`, else the copy
    /// vendored with the model crate's tests.
    fn golden_dir() -> PathBuf {
        std::env::var(super::harness::GOLDEN_ENV)
            .ok()
            .filter(|v| !v.trim().is_empty())
            .map_or_else(
                || {
                    Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("../oxibonsai-model/tests/fixtures/bonsai2_golden")
                },
                PathBuf::from,
            )
    }

    /// Tokens per second over `elapsed`.
    fn rate(tokens: usize, elapsed: Duration) -> f64 {
        tokens as f64 / elapsed.as_secs_f64().max(1e-9)
    }

    /// Bytes as GiB with two decimals.
    fn gib(bytes: u64) -> String {
        format!("{:.2} GiB", bytes as f64 / f64::from(1u32 << 30))
    }

    /// Bytes as MiB with one decimal.
    fn mib(bytes: u64) -> String {
        format!("{:.1} MiB", bytes as f64 / f64::from(1u32 << 20))
    }

    /// Signed bytes as MiB with one decimal.
    fn mib_delta(after: u64, before: u64) -> String {
        let delta = after as f64 - before as f64;
        format!("{:+.1} MiB", delta / f64::from(1u32 << 20))
    }

    /// The process's peak resident set so far (`getrusage`, bytes on
    /// macOS) — what `/usr/bin/time -l` prints as "maximum resident set
    /// size". Process-wide: it includes whatever ran earlier in this binary,
    /// mapped weight pages included.
    fn peak_rss_bytes() -> u64 {
        // SAFETY: `getrusage` fills the zeroed, caller-owned `rusage`.
        let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
        // SAFETY: `RUSAGE_SELF` with a valid out pointer.
        let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
        if rc == 0 {
            u64::try_from(usage.ru_maxrss).unwrap_or(0)
        } else {
            0
        }
    }

    /// The process's memory as the kernel's footprint ledger charges it
    /// (`task_info(TASK_VM_INFO)`). Unlike the resident set it leaves out
    /// clean file-backed pages — the mapped weights, which come and go with
    /// page-cache pressure — and it counts Metal allocations, so its deltas
    /// are what a runner or a CPU model really adds.
    #[derive(Clone, Copy)]
    struct Footprint {
        /// `phys_footprint`: dirty anonymous memory, compressed pages and
        /// device allocations charged to the process.
        phys: u64,
        /// `ledger_tag_graphics_footprint`: the GPU allocations among them.
        graphics: u64,
        /// `ledger_phys_footprint_peak`: the lifetime peak of `phys`.
        peak: u64,
    }

    fn footprint() -> Footprint {
        use mach2::kern_return::KERN_SUCCESS;
        use mach2::message::mach_msg_type_number_t;
        use mach2::task::task_info;
        use mach2::task_info::{task_vm_info, TASK_VM_INFO};
        use mach2::traps::mach_task_self;
        use mach2::vm_types::natural_t;

        let mut info = task_vm_info::default();
        let mut count = mach_msg_type_number_t::try_from(
            std::mem::size_of::<task_vm_info>() / std::mem::size_of::<natural_t>(),
        )
        .unwrap_or(0);
        // SAFETY: `info` is a caller-owned `task_vm_info` and `count` is its
        // size in `natural_t` words, so the kernel writes only inside it.
        let rc = unsafe {
            task_info(
                mach_task_self(),
                TASK_VM_INFO,
                std::ptr::from_mut(&mut info).cast(),
                &mut count,
            )
        };
        if rc != KERN_SUCCESS {
            return Footprint {
                phys: 0,
                graphics: 0,
                peak: 0,
            };
        }
        // Copies out of the packed struct (no references to its fields).
        let phys = info.phys_footprint;
        let graphics = info.ledger_tag_graphics_footprint;
        let peak = info.ledger_phys_footprint_peak;
        Footprint {
            phys,
            graphics: u64::try_from(graphics).unwrap_or(0),
            peak: u64::try_from(peak).unwrap_or(0),
        }
    }

    /// Bytes of a runner's no-copy view of the mapped file: the whole GGUF
    /// image rounded up to the host page. `currentAllocatedSize` counts it
    /// once per runner, although every runner reads the same file pages.
    fn mapped_view_bytes(gguf: &GgufFile<'_>) -> u64 {
        // SAFETY: `sysconf` has no preconditions.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        let page = u64::try_from(page).unwrap_or(16_384).max(1);
        (gguf.data.len() as u64).div_ceil(page) * page
    }

    /// One release band of the 27B.
    pub(super) struct Band {
        quant: &'static str,
        env: &'static str,
        file: &'static str,
    }

    pub(super) const PQ2_0: Band = Band {
        quant: "PQ2_0",
        env: PQ2_ENV,
        file: PQ2_FILE,
    };
    pub(super) const PTQ1_0: Band = Band {
        quant: "PTQ1_0",
        env: PTQ1_ENV,
        file: PTQ1_FILE,
    };

    /// KV window of every engine here: 256 MiB of runner KV, far above the
    /// longest prompt plus its continuation.
    const MAX_SEQ: usize = 4096;
    /// Greedy steps the fork's server dump carries per prompt.
    const GOLDEN_STEPS: usize = 24;
    /// Tokens timed per G10 decode run.
    const G10_TOKENS: usize = 32;
    /// G10's bar: decode tokens per second on the 24 GB M3.
    const G10_MIN_TOK_S: f64 = 5.0;
    /// Tokens of the timed long prefill.
    const LONG_PREFILL: usize = 512;
    /// Keeping the CPU model beside the runner must cost less than this, or
    /// its KV/recurrent buffers would have to be freed while the runner is
    /// active.
    const CPU_MODEL_RSS_CEILING: u64 = 1 << 30;

    /// One greedy run through an engine: the ids, the prefill and decode
    /// times, and (for the Metal engine) every step's G4 comparison.
    struct Run {
        ids: Vec<u32>,
        prefill: Duration,
        decode: Duration,
        worst_delta: f64,
        failures: Vec<String>,
    }

    /// Prefill `tokens` and decode `steps` greedy tokens through the
    /// engine's own seam (`prefill_from_pos` / `decode_step`, what
    /// `generate` runs), comparing every step with `golden` when given.
    fn drive(
        engine: &mut InferenceEngine<'_>,
        tokens: &[u32],
        steps: usize,
        golden: Option<&[super::harness::GoldenStep]>,
    ) -> Run {
        engine.reset();
        let started = Instant::now();
        let mut logits = engine.prefill_from_pos(tokens, 0).expect("prefill");
        let prefill = started.elapsed();
        let started = Instant::now();
        let mut ids = Vec::with_capacity(steps);
        let mut worst_delta = 0.0f64;
        let mut failures = Vec::new();
        for step in 0..steps {
            let id = u32::try_from(argmax(&logits)).expect("token id");
            ids.push(id);
            if let Some(golden_step) = golden.and_then(|g| g.get(step)) {
                let cmp = compare_step(&log_softmax(&logits), id, golden_step);
                worst_delta = worst_delta.max(cmp.worst_delta);
                if cmp.ours_id != cmp.golden_id {
                    failures.push(format!(
                        "step {step}: greedy id {} vs the fork's {}",
                        cmp.ours_id, cmp.golden_id
                    ));
                }
                if !cmp.same_set {
                    failures.push(format!(
                        "step {step}: top-10 set differs (missing {:?}, extra {:?})",
                        cmp.missing, cmp.extra
                    ));
                }
                if cmp.worst_delta > G4_LOGPROB_BAND {
                    failures.push(format!(
                        "step {step}: worst |dlogprob| {:.3e} > {G4_LOGPROB_BAND:e} (rank {}, id \
                         {})",
                        cmp.worst_delta, cmp.worst_at.0, cmp.worst_at.1
                    ));
                }
            }
            if step + 1 < steps {
                logits = engine.decode_step(id, tokens.len() + step).expect("decode");
            }
        }
        Run {
            ids,
            prefill,
            decode: started.elapsed(),
            worst_delta,
            failures,
        }
    }

    /// One golden prompt: its tokens (checked against the fork's dump), the
    /// fork server's 24 steps and its content.
    struct Case {
        tokens: Vec<u32>,
        steps: Vec<super::harness::GoldenStep>,
        content: String,
    }

    fn cases(tokenizer: &TokenizerBridge, quant: &str) -> Vec<Case> {
        let dir = golden_dir();
        PROMPTS
            .iter()
            .enumerate()
            .map(|(index, prompt)| {
                let n = index + 1;
                let tokens = tokenizer.encode(prompt).expect("prompt encodes");
                let dump = super::harness::read_golden(
                    &dir,
                    &format!("Ternary-Bonsai-2-27B-{quant}.prompt{n}.prompt_tokens.txt"),
                );
                assert_eq!(
                    tokens,
                    parse_prompt_tokens(&dump),
                    "prompt {n}: the fork's prompt tokens"
                );
                let json =
                    super::harness::read_golden(&dir, &format!("PQ2_0.prompt{n}.server.json"));
                let steps = parse_golden_steps(&json);
                assert_eq!(steps.len(), GOLDEN_STEPS, "prompt {n}: golden length");
                let root: serde_json::Value =
                    serde_json::from_str(&json).expect("golden JSON parses");
                let content = root["content"]
                    .as_str()
                    .expect("golden carries content")
                    .to_string();
                Case {
                    tokens,
                    steps,
                    content,
                }
            })
            .collect()
    }

    fn engine<'a>(gguf: &'a GgufFile<'a>, backend: Backend) -> InferenceEngine<'a> {
        let engine = InferenceEngine::from_gguf_with_backend(
            gguf,
            greedy_params(GOLDEN_STEPS),
            42,
            MAX_SEQ,
            backend,
        )
        .unwrap_or_else(|e| panic!("the 27B loads on {backend}: {e}"));
        let expected = if backend == Backend::Metal {
            HybridBackend::Metal
        } else {
            HybridBackend::Cpu
        };
        assert_eq!(engine.hybrid_backend(), Some(expected));
        engine
    }

    /// Decode steps per round of [`decode_footprint_is_flat`] (after a
    /// one-step prefill).
    const FLAT_DECODE_STEPS: usize = 30;
    /// Measured rounds: `50 × 31 = 1550` forwards.
    const FLAT_ROUNDS: usize = 50;
    /// Growth the measured forwards may add: an undrained command buffer and
    /// encoder cost about 1.8 KiB per forward, i.e. ~2.7 MiB over the
    /// measured rounds.
    const FLAT_GROWTH_CEILING: u64 = 1 << 20;

    /// The synthetic fixture on the Metal runner, decoding round after round:
    /// once warm, the process footprint must stay flat.
    pub(super) fn decode_footprint_is_flat() {
        let bytes = oxibonsai_testkit::qwen35_fixture::synthetic_qwen35_gguf();
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let mut engine = InferenceEngine::from_gguf_with_backend(
            &gguf,
            greedy_params(8),
            42,
            64,
            Backend::Metal,
        )
        .expect("the fixture builds on the Metal runner");
        assert_eq!(engine.hybrid_backend(), Some(HybridBackend::Metal));
        let prompt = [7u32, 11, 13];
        let run = |engine: &mut InferenceEngine<'_>, rounds: usize| {
            for _ in 0..rounds {
                engine.reset();
                let _ = engine.prefill_from_pos(&prompt, 0).expect("prefill");
                for step in 0..FLAT_DECODE_STEPS {
                    let _ = engine.decode_step(5, prompt.len() + step).expect("decode");
                }
            }
        };
        // Warm-up: pipelines, scratch and allocator high-water marks.
        run(&mut engine, 10);
        let before = footprint();
        assert!(before.phys > 0, "task_info(TASK_VM_INFO) must report");
        run(&mut engine, FLAT_ROUNDS);
        let after = footprint();
        let forwards = FLAT_ROUNDS * (FLAT_DECODE_STEPS + 1);
        eprintln!(
            "{forwards} Metal hybrid forwards: footprint {} (graphics {})",
            mib_delta(after.phys, before.phys),
            mib_delta(after.graphics, before.graphics)
        );
        assert!(
            after.phys.saturating_sub(before.phys) < FLAT_GROWTH_CEILING,
            "{forwards} forwards grew the process footprint by {}: something the runner \
             creates per call outlives it",
            mib_delta(after.phys, before.phys)
        );
    }

    pub(super) fn g11_engine(band: &Band, test: &str) {
        let Some(path) = locate(band.env, band.file, test) else {
            return;
        };
        let gate_start = Instant::now();
        let quant = band.quant;
        let mmap = mmap_gguf_file(&path).expect("27B GGUF maps");
        let gguf = GgufFile::parse(&mmap).expect("27B GGUF parses");
        let tokenizer = tokenizer_from_gguf(&gguf).expect("GGUF-embedded tokenizer");
        let cases = cases(&tokenizer, quant);

        // The CPU engine first (`--backend cpu`), then drop it.
        let mut cpu_runs = Vec::new();
        {
            let mut cpu = engine(&gguf, Backend::Cpu);
            eprintln!("[{quant}] CPU engine: {}", cpu.effective_tier_reason());
            for case in &cases {
                cpu_runs.push(drive(&mut cpu, &case.tokens, GOLDEN_STEPS, None));
            }
        }

        let mut metal = engine(&gguf, Backend::Metal);
        eprintln!(
            "[{quant}] Metal engine: {} | {}",
            metal.kernel_label(),
            metal.effective_tier_reason()
        );
        let mut failures = Vec::new();
        for (index, (case, cpu_run)) in cases.iter().zip(&cpu_runs).enumerate() {
            let n = index + 1;
            let run = drive(&mut metal, &case.tokens, GOLDEN_STEPS, Some(&case.steps));
            let metal_text = tokenizer.decode(&run.ids).expect("decodes");
            let cpu_text = tokenizer.decode(&cpu_run.ids).expect("decodes");
            eprintln!(
                "[{quant}] prompt {n}: {} prompt tokens | CPU prefill {:.2}s ({:.2} tok/s), decode \
                 {:.2} tok/s | Metal prefill {:.2}s ({:.2} tok/s), decode {:.2} tok/s | worst \
                 top-10 |dlogprob| vs the fork {:.3e} | {metal_text:?}",
                case.tokens.len(),
                cpu_run.prefill.as_secs_f64(),
                rate(case.tokens.len(), cpu_run.prefill),
                rate(GOLDEN_STEPS - 1, cpu_run.decode),
                run.prefill.as_secs_f64(),
                rate(case.tokens.len(), run.prefill),
                rate(GOLDEN_STEPS - 1, run.decode),
                run.worst_delta,
            );
            failures.extend(run.failures.iter().map(|f| format!("prompt {n}: {f}")));
            if run.ids != cpu_run.ids {
                failures.push(format!(
                    "prompt {n}: Metal ids {:?} differ from the CPU engine's {:?}",
                    run.ids, cpu_run.ids
                ));
            }
            if metal_text != cpu_text || metal_text != case.content {
                failures.push(format!(
                    "prompt {n}: texts differ: Metal {metal_text:?}, CPU {cpu_text:?}, fork {:?}",
                    case.content
                ));
            }
            // The product path decodes the same ids.
            metal.reset();
            let generated = metal
                .generate(&case.tokens, GOLDEN_STEPS)
                .expect("generate");
            if generated != run.ids {
                failures.push(format!(
                    "prompt {n}: generate {generated:?} differs from the stepwise drive {:?}",
                    run.ids
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "[{quant}] Metal engine vs the CPU engine and the fork:\n{}",
            failures.join("\n")
        );
        record(true, test, Some(gate_start.elapsed()));
    }

    /// A prompt of exactly [`LONG_PREFILL`] ids: ordinary prose, tokenised,
    /// repeated as needed.
    fn long_prompt(tokenizer: &TokenizerBridge) -> Vec<u32> {
        let text = "The history of the harbour town is a long one. Fishing boats left before \
                    dawn, merchants argued over the price of salt, and children ran along the \
                    sea wall while the tide came in. ";
        let piece = tokenizer.encode(text).expect("prose encodes");
        assert!(!piece.is_empty());
        piece.iter().copied().cycle().take(LONG_PREFILL).collect()
    }

    pub(super) fn g10_engine(test: &str) {
        use oxibonsai_runtime::memory::get_rss_bytes;

        let mut ran = 0usize;
        let gate_start = Instant::now();
        for band in [&PQ2_0, &PTQ1_0] {
            let Some(path) = locate(band.env, band.file, test) else {
                continue;
            };
            let quant = band.quant;
            let rss_start = get_rss_bytes();
            let fp_start = footprint();
            let mmap = mmap_gguf_file(&path).expect("27B GGUF maps");
            let gguf = GgufFile::parse(&mmap).expect("27B GGUF parses");
            let view = mapped_view_bytes(&gguf);
            let tokenizer = tokenizer_from_gguf(&gguf).expect("GGUF-embedded tokenizer");
            let load = Instant::now();
            let mut metal = engine(&gguf, Backend::Metal);
            let load = load.elapsed();
            let rss_loaded = get_rss_bytes();
            let fp_loaded = footprint();
            let window = metal
                .hybrid_metal_window()
                .expect("a Metal engine has a window")
                .clone();
            assert_eq!(window.window, MAX_SEQ);
            assert_eq!(
                metal.hybrid_metal_weights_mapped(),
                Some(true),
                "a mapped GGUF is read in place"
            );
            let prompt = tokenizer.encode(PROMPTS[0]).expect("prompt encodes");

            // Warm-up, then the first-token latency of a fresh sequence.
            let _ = drive(&mut metal, &prompt, 4, None);
            metal.reset();
            let started = Instant::now();
            let first_logits = metal.prefill_from_pos(&prompt, 0).expect("prefill");
            let first = u32::try_from(argmax(&first_logits)).expect("token id");
            let first_token_latency = started.elapsed();

            // Decode: best of three 32-token runs.
            let mut runs = Vec::new();
            for _ in 0..3 {
                metal.reset();
                let logits = metal.prefill_from_pos(&prompt, 0).expect("prefill");
                let mut next = u32::try_from(argmax(&logits)).expect("token id");
                let started = Instant::now();
                for step in 0..G10_TOKENS {
                    let row = metal
                        .decode_step(next, prompt.len() + step)
                        .expect("decode");
                    next = u32::try_from(argmax(&row)).expect("token id");
                }
                runs.push(rate(G10_TOKENS, started.elapsed()));
            }
            let best = runs.iter().copied().fold(0.0f64, f64::max);

            // A 512-token prefill.
            let long = long_prompt(&tokenizer);
            metal.reset();
            let started = Instant::now();
            let _ = metal.prefill_from_pos(&long, 0).expect("long prefill");
            let long_prefill = started.elapsed();
            let rss_decoded = get_rss_bytes();
            let fp_decoded = footprint();
            // The long prefill has grown the scratch to a full chunk: this
            // is the runner at its largest.
            let device_first = metal
                .hybrid_metal_device_allocated_bytes()
                .expect("device bytes");

            // What keeping the CPU model beside the runner costs: one more
            // CPU model bound at the same window.
            let rss_before_cpu_model = get_rss_bytes();
            let fp_before_cpu_model = footprint();
            let cpu_model = {
                let config =
                    oxibonsai_model::hybrid::HybridModel::config_from_gguf(&gguf).expect("config");
                let kernel = std::sync::Arc::new(oxibonsai_kernels::KernelDispatcher::with_tier(
                    oxibonsai_kernels::cpu_kernel_tier(),
                ));
                oxibonsai_model::hybrid::HybridModel::from_gguf_with(
                    &gguf, config, MAX_SEQ, &kernel,
                )
                .expect("a second CPU model binds")
            };
            let rss_after_cpu_model = get_rss_bytes();
            let fp_after_cpu_model = footprint();
            let cpu_model_cost = rss_after_cpu_model.saturating_sub(rss_before_cpu_model);
            drop(cpu_model);

            // A second runner in the same process: its own session, KV,
            // state and scratch over the same mapped pages.
            let rss_before_second = get_rss_bytes();
            let fp_before_second = footprint();
            let mut second = engine(&gguf, Backend::Metal);
            let second_logits = second.prefill_from_pos(&prompt, 0).expect("second prefill");
            assert_eq!(
                u32::try_from(argmax(&second_logits)).expect("token id"),
                first,
                "a second runner decodes the same first token"
            );
            let rss_after_second = get_rss_bytes();
            let fp_after_second = footprint();
            let device_both = second
                .hybrid_metal_device_allocated_bytes()
                .expect("device bytes");
            drop(second);
            let device_second = device_both.saturating_sub(device_first);

            eprintln!(
                "G10-engine[{quant}] speed: decode best {best:.2} tok/s (runs {}) | prefill \
                 {LONG_PREFILL} tokens {:.2}s = {:.2} tok/s | first-token latency {:.2}s ({} \
                 prompt tokens) | load {:.1}s | window {} ({})",
                runs.iter()
                    .map(|r| format!("{r:.2}"))
                    .collect::<Vec<_>>()
                    .join(", "),
                long_prefill.as_secs_f64(),
                rate(LONG_PREFILL, long_prefill),
                first_token_latency.as_secs_f64(),
                prompt.len(),
                load.as_secs_f64(),
                window.window,
                window.describe_limits(),
            );
            eprintln!(
                "G10-engine[{quant}] memory: RSS {} after load, {} after decode, process peak {} \
                 | footprint {} after load, {} after decode, of it graphics {}; lifetime \
                 footprint peak {} | runner device allocation {} = no-copy view of the mapped \
                 file {} + own buffers {} (planned {} with a full prefill chunk of scratch)",
                mib_delta(rss_loaded, rss_start),
                mib_delta(rss_decoded, rss_start),
                gib(peak_rss_bytes()),
                mib_delta(fp_loaded.phys, fp_start.phys),
                mib_delta(fp_decoded.phys, fp_start.phys),
                mib_delta(fp_decoded.graphics, fp_start.graphics),
                gib(fp_decoded.peak),
                mib(device_first),
                mib(view),
                mib_delta(device_first, view),
                mib(window.runner_allocated_bytes),
            );
            eprintln!(
                "G10-engine[{quant}] residents: keeping the CPU model costs {} RSS, {} footprint \
                 | a second runner adds {} RSS, {} footprint (graphics {}) and {} of device \
                 allocation = no-copy view {} (the same file pages, resident once) + own buffers \
                 {} (its scratch sized by a {}-token prefill)",
                mib_delta(rss_after_cpu_model, rss_before_cpu_model),
                mib_delta(fp_after_cpu_model.phys, fp_before_cpu_model.phys),
                mib_delta(rss_after_second, rss_before_second),
                mib_delta(fp_after_second.phys, fp_before_second.phys),
                mib_delta(fp_after_second.graphics, fp_before_second.graphics),
                mib(device_second),
                mib(view),
                mib_delta(device_second, view),
                prompt.len(),
            );
            assert!(
                best >= G10_MIN_TOK_S,
                "[{quant}] Metal engine decode {best:.2} tok/s < {G10_MIN_TOK_S}"
            );
            assert!(
                cpu_model_cost < CPU_MODEL_RSS_CEILING,
                "[{quant}] keeping the CPU model costs {} of RSS (>= 1 GiB): free its KV and \
                 recurrent buffers while the runner is active",
                gib(cpu_model_cost)
            );
            ran += 1;
        }
        if ran == 2 {
            record(true, test, Some(gate_start.elapsed()));
        } else {
            eprintln!("{test}: {ran} of 2 bands located; not recorded as executed");
        }
    }

    #[cfg(feature = "server")]
    pub(super) fn server_round_trip(test: &str) {
        let mut ran = 0usize;
        let gate_start = Instant::now();
        for band in [&PQ2_0, &PTQ1_0] {
            let Some(path) = locate(band.env, band.file, test) else {
                continue;
            };
            let quant = band.quant;
            // One mapping per band, shared by both engines: `create_router`
            // takes a `'static` engine, so the mapping is leaked for the rest
            // of the process — once per band, not once per engine.
            let gguf = leak_mapping(&path);
            let cpu = chat_once(gguf, Backend::Cpu);
            let metal = chat_once(gguf, Backend::Metal);
            eprintln!("[{quant}] /v1/chat/completions: CPU {cpu} | Metal {metal}");
            assert_eq!(
                metal, cpu,
                "[{quant}] the Metal engine's chat message must equal the CPU engine's"
            );
            assert!(
                metal["content"].as_str().is_some_and(|c| !c.is_empty())
                    || metal["reasoning_content"]
                        .as_str()
                        .is_some_and(|c| !c.is_empty()),
                "[{quant}] an empty answer: {metal}"
            );
            ran += 1;
        }
        if ran == 2 {
            record(true, test, Some(gate_start.elapsed()));
        } else {
            eprintln!("{test}: {ran} of 2 bands located; not recorded as executed");
        }
    }

    /// Map and parse `path` for the rest of the process (what a `'static`
    /// engine borrows).
    #[cfg(feature = "server")]
    fn leak_mapping(path: &Path) -> &'static GgufFile<'static> {
        let mmap = mmap_gguf_file(path).expect("27B GGUF maps");
        let mmap: &'static memmap2::Mmap = Box::leak(Box::new(mmap));
        Box::leak(Box::new(GgufFile::parse(mmap).expect("27B GGUF parses")))
    }

    /// One `/v1/chat/completions` request, temperature 0, on a router over
    /// a `backend` engine of `gguf`; returns `choices[0].message`. The
    /// router, and with it the engine, is dropped before this returns.
    #[cfg(feature = "server")]
    fn chat_once(gguf: &'static GgufFile<'static>, backend: Backend) -> serde_json::Value {
        use axum::body::Body;
        use axum::http::{Request, StatusCode};
        use tower::ServiceExt;

        let engine =
            InferenceEngine::from_gguf_with_backend(gguf, greedy_params(16), 42, MAX_SEQ, backend)
                .unwrap_or_else(|e| panic!("the 27B loads on {backend}: {e}"));
        let tokenizer = TokenizerBridge::native_from_gguf_metadata(&gguf.metadata)
            .expect("GGUF-embedded tokenizer with its chat template");
        let router = oxibonsai_runtime::server::create_router(engine, Some(tokenizer));
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("tokio runtime");
        rt.block_on(async {
            let body = serde_json::json!({
                "model": "bonsai2",
                "messages": [
                    { "role": "user", "content": "What is the capital of Japan? Answer in one word." }
                ],
                "max_tokens": 16,
                "temperature": 0.0
            });
            let request = Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .expect("request");
            let response = router.oneshot(request).await.expect("response");
            assert_eq!(response.status(), StatusCode::OK);
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .expect("body");
            let json: serde_json::Value = serde_json::from_slice(&bytes).expect("JSON");
            json["choices"][0]["message"].clone()
        })
    }
}
