//! The Metal hybrid runner's batched prefill on the real Bonsai 2 27B, both
//! bands (`PQ2_0`, `PTQ1_0`).
//!
//! * `hybrid_real_27b_metal_prefill_throughput_bonsai2` — a 512-token
//!   prefill in the batched mode (every projection a tiled GEMM) must cost
//!   at most a third of a decode token, per token: the prefill's ms/token
//!   against the best of three 16-token decode runs of the same process.
//!   Printed beside it: the batched prefill at 64 / 128 / 256 / 512 / 1024
//!   tokens, the sequential (decode-rate) prefill at 512, and the load
//!   average. With `OXI_BONSAI2_PREFILL_SWEEP=1` the sequential prefill is
//!   also timed at every size.
//! * `hybrid_real_27b_metal_batched_prefill_matches_cpu_tokens_bonsai2` —
//!   design §8.2 G11 with the batched prefill on, the GEMM threshold lowered
//!   to 2 so every golden prompt's prefill runs on the GEMM (the three
//!   prompts are 5–13 tokens, below the default threshold of 16): 32 greedy
//!   steps × 3 prompts, the runner's token equal to the CPU model's at every
//!   step (both fed the CPU's token) and both texts equal to the fork's; and
//!   the G4 logprob band against the fork's Metal golden
//!   (`PQ2_0.prompt{i}.server.json`, 24 steps with top-10 logprobs): the
//!   runner's top-1 equals the golden's at every step and the worst
//!   `|Δlogprob|` over the golden top-10 stays within 2e-2. Every step's
//!   golden top-1/top-2 gap is printed, so the known near-tie (prompt 3,
//!   step 20, 0.0165 nats) is diagnosed with its numbers rather than
//!   assumed.
//!
//! # Where the files come from
//!
//! Only from `OXI_BONSAI2_PQ2_GGUF` and `OXI_BONSAI2_PTQ1_GGUF` — no
//! models-directory fallback, so a workspace test run never maps a 27B by
//! accident — and the goldens from `OXI_BONSAI2_GOLDEN_DIR` (else the
//! vendored copy). A gate whose files are not set skips with a
//! `bonsai2-models` capability record; `OXI_REQUIRE_MODEL_FILES=1` turns
//! that into a failure. A gate that ran both bands records `executed`.
//!
//! # Memory
//!
//! One band at a time; the runner reads the weights in place from the same
//! file mapping the CPU model borrows.

#[path = "bonsai2_real/harness.rs"]
mod harness;

/// Capability-record name of the throughput gate.
const THROUGHPUT_TEST: &str =
    "oxibonsai-model::hybrid_metal_prefill_gates::hybrid_real_27b_metal_prefill_throughput_bonsai2";
/// Capability-record name of the batched G11 / G4-band gate.
const TOKENS_TEST: &str = "oxibonsai-model::hybrid_metal_prefill_gates::hybrid_real_27b_metal_batched_prefill_matches_cpu_tokens_bonsai2";

/// Serialises the gates of this binary (one 27B mapping at a time).
static GATE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Holds [`GATE_LOCK`] and keeps `OXIBONSAI_KERNEL_TIER` cleared for one
/// gate, restoring its prior value on drop: the CPU model's `PQ2_0` GEMV
/// honours that INT8 selector on any tier (K-14) while the Metal runner
/// never reads it, so an exported value would flip the CPU-vs-Metal
/// comparison.
struct TierEnvGuard {
    _lock: std::sync::MutexGuard<'static, ()>,
    prior: Option<String>,
}

impl TierEnvGuard {
    fn cleared() -> Self {
        let lock = GATE_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let prior = std::env::var(oxibonsai_kernels::dispatch_int8::KERNEL_TIER_ENV).ok();
        // SAFETY: `lock` is held for the guard's lifetime and serialises
        // every reader and writer of the environment in this binary.
        unsafe { std::env::remove_var(oxibonsai_kernels::dispatch_int8::KERNEL_TIER_ENV) };
        Self { _lock: lock, prior }
    }
}

impl Drop for TierEnvGuard {
    fn drop(&mut self) {
        let key = oxibonsai_kernels::dispatch_int8::KERNEL_TIER_ENV;
        // SAFETY: still under `GATE_LOCK` (see `cleared`).
        unsafe {
            match &self.prior {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
    }
}

/// The batched prefill runs at least three times the decode rate, per band.
#[test]
fn hybrid_real_27b_metal_prefill_throughput_bonsai2() {
    let _env = TierEnvGuard::cleared();
    #[cfg(all(feature = "metal", target_os = "macos"))]
    gates::throughput(THROUGHPUT_TEST);
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    skip_without_metal(THROUGHPUT_TEST);
}

/// G11 and the G4 logprob band with every prefill on the GEMM, per band.
#[test]
fn hybrid_real_27b_metal_batched_prefill_matches_cpu_tokens_bonsai2() {
    let _env = TierEnvGuard::cleared();
    #[cfg(all(feature = "metal", target_os = "macos"))]
    gates::tokens(TOKENS_TEST);
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    skip_without_metal(TOKENS_TEST);
}

/// A build without the Metal backend cannot run either gate.
#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn skip_without_metal(test: &str) {
    assert!(
        !harness::require_model_files(),
        "{}=1 but this test binary was built without the `metal` feature on macOS",
        harness::REQUIRE_ENV
    );
    eprintln!("skip {test}: the Metal backend is not compiled in");
    harness::record_capability(false, test);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod gates {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::time::Instant;

    use oxibonsai_core::gguf::reader::{mmap_gguf_file, GgufFile};
    use oxibonsai_kernels::gpu_backend::metal_full_layer::qwen35::Qwen35PrefillMode;
    use oxibonsai_kernels::{cpu_kernel_tier, KernelDispatcher};
    use oxibonsai_model::hybrid::metal::HybridMetalRunner;
    use oxibonsai_model::hybrid::model::HybridModel;

    use crate::harness::{
        argmax, compare_step, detokenize, gguf_vocab, golden_dir, gpt2_byte_decoder, log_softmax,
        parse_golden_steps, parse_prompt_tokens, read_golden, read_golden_bytes,
        record_capability_timed, require_model_files, show_bytes, top_n, G4_LOGPROB_BAND, PQ2_ENV,
        PROMPTS, PTQ1_ENV, REQUIRE_ENV, TOP_N,
    };

    /// KV window: the 1024-token prefill plus decode, with room to spare.
    const MAX_SEQ: usize = 2048;
    /// Tokens each golden continuation holds (`llama-cli -n 32`).
    const TEXT_STEPS: usize = 32;
    /// Steps the fork's server dump carries per prompt.
    const GOLDEN_STEPS: usize = 24;
    /// `llama-cli` appends this to every continuation dump.
    const TEXT_TRAILER: &[u8] = b"\n\n";
    /// Tokens of the timed prefill the gate asserts on.
    const GATE_PREFILL: usize = 512;
    /// Every prefill size the throughput gate prints.
    const SWEEP: [usize; 5] = [64, 128, 256, 512, 1024];
    /// Tokens per decode run of the throughput gate.
    const DECODE_TOKENS: usize = 16;
    /// The prefill must run at least this many times the decode rate.
    const MIN_SPEEDUP: f64 = 3.0;
    /// Opt-in for the sequential prefill at every sweep size.
    const SWEEP_ENV: &str = "OXI_BONSAI2_PREFILL_SWEEP";

    /// One release band of the 27B.
    struct Band {
        label: &'static str,
        env: &'static str,
        golden_prefix: &'static str,
    }

    const BANDS: [Band; 2] = [
        Band {
            label: "PQ2_0",
            env: PQ2_ENV,
            golden_prefix: "Ternary-Bonsai-2-27B-PQ2_0",
        },
        Band {
            label: "PTQ1_0",
            env: PTQ1_ENV,
            golden_prefix: "Ternary-Bonsai-2-27B-PTQ1_0",
        },
    ];

    /// The bands whose file is named by its environment variable and
    /// exists. Panics under `OXI_REQUIRE_MODEL_FILES=1` when any is missing.
    fn locate_bands(test: &str) -> Vec<(&'static Band, PathBuf)> {
        let mut found = Vec::new();
        for band in &BANDS {
            let path = std::env::var(band.env)
                .ok()
                .filter(|v| !v.trim().is_empty())
                .map(PathBuf::from);
            match path {
                Some(p) if p.is_file() => found.push((band, p)),
                other => {
                    assert!(
                        !require_model_files(),
                        "{REQUIRE_ENV}=1 but {} does not name an existing file ({other:?})",
                        band.env
                    );
                    eprintln!(
                        "{test}: {} not set to an existing file; band skipped",
                        band.env
                    );
                }
            }
        }
        found
    }

    /// Record the outcome: `executed` only when every band ran.
    fn record(test: &str, ran: usize, started: Instant) {
        if ran == BANDS.len() {
            record_capability_timed(true, test, Some(started.elapsed()));
        } else {
            eprintln!("skip {test}: {ran} of {} bands located", BANDS.len());
            record_capability_timed(false, test, None);
        }
    }

    /// The machine's 1/5/15-minute load average, for the record.
    fn load_average() -> String {
        std::process::Command::new("sysctl")
            .args(["-n", "vm.loadavg"])
            .output()
            .ok()
            .and_then(|out| String::from_utf8(out.stdout).ok())
            .map_or_else(|| "unavailable".to_string(), |s| s.trim().to_string())
    }

    fn load_cpu<'a>(gguf: &'a GgufFile<'a>, label: &str) -> HybridModel<'a> {
        let config = HybridModel::config_from_gguf(gguf).expect("27B config");
        let kernel = Arc::new(KernelDispatcher::with_tier(cpu_kernel_tier()));
        let t0 = Instant::now();
        let model = HybridModel::from_gguf_with(gguf, config, MAX_SEQ, &kernel)
            .unwrap_or_else(|e| panic!("{label}: the CPU model loads: {e}"));
        eprintln!(
            "{label}: CPU model loaded in {:.2}s",
            t0.elapsed().as_secs_f64()
        );
        model
    }

    fn load_runner<'a>(
        model: &HybridModel<'a>,
        mmap: &'a memmap2::Mmap,
        label: &str,
    ) -> HybridMetalRunner<'a> {
        let t0 = Instant::now();
        let runner = HybridMetalRunner::new_mapped(model, mmap)
            .unwrap_or_else(|e| panic!("{label}: the Metal runner builds: {e}"));
        assert!(runner.is_mapped(), "{label}: weights must be read in place");
        assert_eq!(runner.prefill_mode(), Qwen35PrefillMode::Batched);
        eprintln!(
            "{label}: Metal runner built in {:.2}s (batch {}, GEMM from {} tokens)",
            t0.elapsed().as_secs_f64(),
            runner.max_batch(),
            runner.gemm_min_cols()
        );
        runner
    }

    fn golden_prompt(band: &Band, prompt_index: usize) -> Vec<u32> {
        let tokens = parse_prompt_tokens(&read_golden(
            &golden_dir(),
            &format!(
                "{}.prompt{prompt_index}.prompt_tokens.txt",
                band.golden_prefix
            ),
        ));
        assert!(
            !tokens.is_empty(),
            "{} prompt {prompt_index}: no tokens",
            band.label
        );
        tokens
    }

    fn golden_text(band: &Band, prompt_index: usize) -> Vec<u8> {
        let mut text = read_golden_bytes(
            &golden_dir(),
            &format!("{}.prompt{prompt_index}.txt", band.golden_prefix),
        );
        if text.ends_with(TEXT_TRAILER) {
            text.truncate(text.len() - TEXT_TRAILER.len());
        }
        text
    }

    /// A deterministic prompt of `n` token ids spread over the vocabulary.
    fn synthetic_prompt(n: usize, vocab: usize) -> Vec<u32> {
        (0..n)
            .map(|i| u32::try_from((i * 7_919 + 13) % (vocab - 1024)).unwrap_or(13))
            .collect()
    }

    // ─────────────────────────────────────────────────────────────────────
    //  Throughput
    // ─────────────────────────────────────────────────────────────────────

    pub(super) fn throughput(test: &str) {
        let started = Instant::now();
        let bands = locate_bands(test);
        if bands.is_empty() {
            record(test, 0, started);
            return;
        }
        let mut failures = Vec::new();
        for (band, path) in &bands {
            failures.extend(throughput_band(band, path));
        }
        assert!(
            failures.is_empty(),
            "prefill throughput:\n{}",
            failures.join("\n")
        );
        record(test, bands.len(), started);
    }

    /// Wall milliseconds per token of one prefill of `tokens` from 0.
    fn prefill_ms_per_token(gpu: &mut HybridMetalRunner<'_>, tokens: &[u32]) -> f64 {
        let mut logits = vec![0.0f32; gpu.vocab_size()];
        gpu.reset();
        let t0 = Instant::now();
        gpu.forward_prefill(tokens, 0, &mut logits)
            .expect("prefill");
        let ms = t0.elapsed().as_secs_f64() * 1e3 / tokens.len() as f64;
        assert!(logits.iter().all(|v| v.is_finite()), "finite logits");
        ms
    }

    fn throughput_band(band: &Band, path: &Path) -> Vec<String> {
        let label = band.label;
        let mmap = mmap_gguf_file(path).unwrap_or_else(|e| panic!("mmap {}: {e}", path.display()));
        let gguf = GgufFile::parse(&mmap).expect("27B GGUF parses");
        let cpu = load_cpu(&gguf, label);
        let mut gpu = load_runner(&cpu, &mmap, label);
        let vocab = gpu.vocab_size();
        let long = synthetic_prompt(*SWEEP.iter().max().unwrap_or(&1024), vocab);

        // Warm-up: pages every weight in, builds every pipeline and grows
        // the activation scratch to a full batch.
        let _ = prefill_ms_per_token(&mut gpu, &long[..GATE_PREFILL]);

        // Decode: best of three 16-token runs after a short prompt.
        let prompt = golden_prompt(band, 1);
        let mut logits = vec![0.0f32; vocab];
        let mut decode_ms = f64::INFINITY;
        for run in 1..=3 {
            gpu.reset();
            gpu.forward_prefill(&prompt, 0, &mut logits)
                .expect("prefill");
            let t0 = Instant::now();
            let mut gpu_seconds = 0.0f64;
            for step in 0..DECODE_TOKENS {
                let token = u32::try_from(argmax(&logits)).expect("token id");
                gpu.forward_into(token, prompt.len() + step, &mut logits)
                    .expect("decode");
                gpu_seconds += gpu.last_gpu_seconds();
            }
            let ms = t0.elapsed().as_secs_f64() * 1e3 / DECODE_TOKENS as f64;
            decode_ms = decode_ms.min(ms);
            eprintln!(
                "prefill gate {label} decode run {run}: {ms:.1} ms/token wall (GPU {:.1} \
                 ms/token); load average {}",
                gpu_seconds * 1e3 / DECODE_TOKENS as f64,
                load_average()
            );
        }

        // Batched prefill at every size; the gate reads 512.
        let mut batched_512 = f64::INFINITY;
        for &n in &SWEEP {
            let ms = prefill_ms_per_token(&mut gpu, &long[..n]);
            if n == GATE_PREFILL {
                batched_512 = ms;
            }
            eprintln!(
                "prefill gate {label} batched prefill {n:>4} tokens: {ms:.1} ms/token ({:.2} s, \
                 {:.2}x the decode rate); load average {}",
                ms * n as f64 / 1e3,
                decode_ms / ms,
                load_average()
            );
        }

        // Sequential (decode-rate) prefill: 512 always, every size opt-in.
        gpu.set_prefill_mode(Qwen35PrefillMode::Sequential);
        let sweep_all = std::env::var(SWEEP_ENV).is_ok_and(|v| v.trim() == "1");
        for &n in &SWEEP {
            if n != GATE_PREFILL && !sweep_all {
                continue;
            }
            let ms = prefill_ms_per_token(&mut gpu, &long[..n]);
            eprintln!(
                "prefill gate {label} sequential prefill {n:>4} tokens: {ms:.1} ms/token ({:.2} \
                 s); load average {}",
                ms * n as f64 / 1e3,
                load_average()
            );
        }
        gpu.set_prefill_mode(Qwen35PrefillMode::Batched);

        let ratio = decode_ms / batched_512;
        eprintln!(
            "prefill gate {label}: {GATE_PREFILL}-token batched prefill {batched_512:.1} ms/token \
             vs best decode {decode_ms:.1} ms/token = {ratio:.2}x (bar {MIN_SPEEDUP}x)"
        );
        if ratio < MIN_SPEEDUP {
            vec![format!(
                "{label}: {GATE_PREFILL}-token prefill {batched_512:.1} ms/token is {ratio:.2}x the \
                 best decode rate {decode_ms:.1} ms/token (bar {MIN_SPEEDUP}x)"
            )]
        } else {
            Vec::new()
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    //  G11 and the G4 logprob band with the batched prefill
    // ─────────────────────────────────────────────────────────────────────

    pub(super) fn tokens(test: &str) {
        let started = Instant::now();
        let bands = locate_bands(test);
        if bands.is_empty() {
            record(test, 0, started);
            return;
        }
        let mut failures = Vec::new();
        for (band, path) in &bands {
            failures.extend(tokens_band(band, path));
        }
        assert!(
            failures.is_empty(),
            "batched G11 / G4 band on the real 27B:\n{}",
            failures.join("\n")
        );
        record(test, bands.len(), started);
    }

    fn tokens_band(band: &Band, path: &Path) -> Vec<String> {
        let label = band.label;
        let mmap = mmap_gguf_file(path).unwrap_or_else(|e| panic!("mmap {}: {e}", path.display()));
        let gguf = GgufFile::parse(&mmap).expect("27B GGUF parses");
        let mut cpu = load_cpu(&gguf, label);
        let mut gpu = load_runner(&cpu, &mmap, label);
        gpu.set_gemm_min_cols(2)
            .expect("every multi-token prefill on the GEMM");
        let vocab_strings = gguf_vocab(&gguf);
        let decoder = gpt2_byte_decoder();
        let vocab = cpu.config().base.vocab_size;
        let (mut cl, mut gl) = (vec![0.0f32; vocab], vec![0.0f32; vocab]);
        let mut failures = Vec::new();
        for prompt_index in 1..=3usize {
            let tokens = golden_prompt(band, prompt_index);
            assert_eq!(
                detokenize(&vocab_strings, &decoder, &tokens),
                PROMPTS[prompt_index - 1].as_bytes(),
                "{label} prompt {prompt_index}: the golden tokens decode to the raw prompt"
            );
            let golden = parse_golden_steps(&read_golden(
                &golden_dir(),
                &format!("PQ2_0.prompt{prompt_index}.server.json"),
            ));
            assert_eq!(golden.len(), GOLDEN_STEPS, "prompt {prompt_index} golden");
            cpu.reset();
            gpu.reset();
            let t0 = Instant::now();
            cpu.forward_prefill(&tokens, 0, &mut cl)
                .unwrap_or_else(|e| panic!("{label} prompt {prompt_index}: CPU prefill: {e}"));
            let cpu_prefill = t0.elapsed().as_secs_f64();
            let t0 = Instant::now();
            gpu.forward_prefill(&tokens, 0, &mut gl)
                .unwrap_or_else(|e| panic!("{label} prompt {prompt_index}: Metal prefill: {e}"));
            let gpu_prefill = t0.elapsed().as_secs_f64();
            eprintln!(
                "batched G11 {label} prompt {prompt_index}: prefill {} tokens on the GEMM: CPU \
                 {cpu_prefill:.2}s, Metal {gpu_prefill:.3}s",
                tokens.len()
            );
            let (mut cpu_tokens, mut gpu_tokens) = (Vec::new(), Vec::new());
            let mut worst_band = (0.0f64, 0usize);
            for step in 0..TEXT_STEPS {
                let c = u32::try_from(argmax(&cl)).expect("token id");
                let g = u32::try_from(argmax(&gl)).expect("token id");
                let (dlp, cpu_gap) = compare_rows(&cl, &gl);
                let mut line = format!(
                    "batched G11 {label} p{prompt_index} step {step:>2}: cpu {c:>6} metal {g:>6} \
                     {} | top-{TOP_N} |dlogprob| metal vs cpu {dlp:.3e} | cpu top-2 gap \
                     {cpu_gap:.4}",
                    if c == g { "ok " } else { "DIFF" }
                );
                if c != g {
                    failures.push(format!(
                        "{label} prompt {prompt_index} step {step}: Metal greedy token {g} != \
                         CPU {c} (CPU top-1/top-2 gap {cpu_gap:.4})"
                    ));
                }
                if let Some(golden_step) = golden.get(step) {
                    let cmp = compare_step(&log_softmax(&gl), g, golden_step);
                    line.push_str(&format!(
                        " | vs fork Metal golden: id {} {} |dlogprob| {:.3e} golden top-2 gap \
                         {:.4}{}",
                        cmp.golden_id,
                        if cmp.ours_id == cmp.golden_id {
                            "ok"
                        } else {
                            "DIFF"
                        },
                        cmp.worst_delta,
                        cmp.golden_gap,
                        if cmp.same_set {
                            ""
                        } else {
                            " (top-10 set differs)"
                        }
                    ));
                    if cmp.worst_delta > worst_band.0 {
                        worst_band = (cmp.worst_delta, step);
                    }
                    if cmp.ours_id != cmp.golden_id {
                        failures.push(format!(
                            "{label} prompt {prompt_index} step {step}: top-1 {} != the fork's \
                             {} (golden top-2 gap {:.4}, |dlogprob| {:.3e})",
                            cmp.ours_id, cmp.golden_id, cmp.golden_gap, cmp.worst_delta
                        ));
                    }
                    if cmp.worst_delta > G4_LOGPROB_BAND {
                        failures.push(format!(
                            "{label} prompt {prompt_index} step {step}: worst |dlogprob| \
                             {:.3e} > {G4_LOGPROB_BAND:e} (rank {}, id {})",
                            cmp.worst_delta, cmp.worst_at.0, cmp.worst_at.1
                        ));
                    }
                }
                eprintln!("{line}");
                cpu_tokens.push(c);
                gpu_tokens.push(g);
                if step + 1 < TEXT_STEPS {
                    let pos = tokens.len() + step;
                    cpu.forward(c, pos, &mut cl)
                        .unwrap_or_else(|e| panic!("{label}: CPU decode at {pos}: {e}"));
                    gpu.forward_into(c, pos, &mut gl)
                        .unwrap_or_else(|e| panic!("{label}: Metal decode at {pos}: {e}"));
                }
            }
            let expected = golden_text(band, prompt_index);
            let cpu_text = detokenize(&vocab_strings, &decoder, &cpu_tokens);
            let gpu_text = detokenize(&vocab_strings, &decoder, &gpu_tokens);
            let equal = cpu_tokens
                .iter()
                .zip(&gpu_tokens)
                .filter(|(a, b)| a == b)
                .count();
            eprintln!(
                "batched G11 {label} prompt {prompt_index}: {equal}/{TEXT_STEPS} greedy tokens \
                 equal (Metal vs CPU); G4-band worst |dlogprob| vs the fork's Metal golden {:.3e} at \
                 step {} (band {G4_LOGPROB_BAND:e}); text CPU {} Metal {}: \"{}\"",
                worst_band.0,
                worst_band.1,
                if cpu_text == expected {
                    "== fork"
                } else {
                    "!= fork"
                },
                if gpu_text == expected {
                    "== fork"
                } else {
                    "!= fork"
                },
                show_bytes(&gpu_text)
            );
            for (who, text) in [("CPU", &cpu_text), ("Metal", &gpu_text)] {
                if *text != expected {
                    failures.push(format!(
                        "{label} prompt {prompt_index}: {who} continuation differs from the \
                         fork's text\n  ours:   \"{}\"\n  golden: \"{}\"",
                        show_bytes(text),
                        show_bytes(&expected)
                    ));
                }
            }
        }
        failures
    }

    /// Worst `|Δlogprob|` over the CPU's top-N ids, and the CPU's top-1/top-2
    /// logprob gap.
    fn compare_rows(cpu: &[f32], gpu: &[f32]) -> (f64, f64) {
        let (lc, lg) = (log_softmax(cpu), log_softmax(gpu));
        let top = top_n(&lc, TOP_N);
        let worst = top
            .iter()
            .map(|&(id, lp)| (lg[id as usize] - lp).abs())
            .fold(0.0f64, f64::max);
        let gap = match (top.first(), top.get(1)) {
            (Some(a), Some(b)) => a.1 - b.1,
            _ => f64::INFINITY,
        };
        (worst, gap)
    }
}
