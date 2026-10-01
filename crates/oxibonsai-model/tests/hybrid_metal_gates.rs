//! Design §8.2 **G10** and **G11** for the Metal hybrid runner
//! ([`oxibonsai_model::hybrid::metal::HybridMetalRunner`]) on the real
//! Bonsai 2 27B, both bands (`PQ2_0`, `PTQ1_0`).
//!
//! * **G11** (`hybrid_real_27b_metal_matches_cpu_tokens_bonsai2`) — first,
//!   every Metal kernel against the CPU SIMD path on **real** activations:
//!   layer 0 (Gated DeltaNet) and layer 3 (the first full-attention layer)
//!   are traced stage by stage on the GPU, and each stage is recomputed on
//!   the CPU from the GPU's own previous stage, so each kernel is measured
//!   alone (cosine ≥ 0.999; the rotations and the embedding bit-for-bit; every
//!   weight decoder bit-for-bit through one-hot GEMV probes). Then greedy
//!   decoding, 32 tokens × the three golden prompts: the runner's greedy
//!   token equals the CPU
//!   [`HybridModel`](oxibonsai_model::hybrid::model::HybridModel)'s at every
//!   step, and both decode to the fork's own 32-token text dump. The
//!   per-step worst top-10 `|Δlogprob|` runner-vs-CPU is printed.
//! * **G10** (`hybrid_real_27b_metal_decode_throughput_bonsai2`) — decode
//!   throughput of the runner, the best of three 32-token runs after a
//!   warm-up, at least 5 tokens/s per band; the load average
//!   and the GPU-side time per token are printed next to it. With
//!   `OXI_BONSAI2_CPU_THROUGHPUT=1` the CPU forward's decode rate (best of
//!   three) is printed beside it for comparison.
//!
//! # Where the files come from
//!
//! Only from `OXI_BONSAI2_PQ2_GGUF` and `OXI_BONSAI2_PTQ1_GGUF` (paths to the
//! two release files) — deliberately no models-directory fallback, so a
//! workspace test run can never map a 27B by accident — and the fork's
//! goldens from `OXI_BONSAI2_GOLDEN_DIR` (else the vendored copy). A gate
//! whose files are not set skips with a `bonsai2-models` capability record;
//! `OXI_REQUIRE_MODEL_FILES=1` turns that into a failure. A gate that ran
//! both bands records `executed`.
//!
//! # Memory
//!
//! One band at a time. The runner reads the weights in place from the same
//! file mapping the CPU model borrows (no-copy Metal buffer), so a band
//! costs the mapping once plus the runner's KV cache (64 KiB/token), its
//! recurrent state (150 MiB) and the CPU model's own caches.
//!
//! # Measured results (24 GB M3, other builds and tests running: 1-minute load average 27–130)
//!
//! | band | G10 best of 3, across 3–4 gate runs | GPU per token | 5-token prefill | G11 tokens | worst top-10 \|Δlogprob\| per prompt |
//! |---|---|---|---|---|---|
//! | `PQ2_0` | 6.44–7.49 tok/s | 131–153 ms | 0.59–0.72 s | 96/96 | 3.1e-4, 4.7e-4, 1.5e-4 |
//! | `PTQ1_0` | 7.01–7.61 tok/s | 126–139 ms | 0.57–0.62 s | 96/96 | 2.8e-4, 5.1e-4, 1.3e-4 |
//!
//! Wall time per token stays within 3 % of the GPU time: decode is GPU
//! bound. On real activations every float stage of layers 0 and 3 has cosine
//! 1.000000000 against the CPU (worst relative error 1.5e-6), and the
//! embedding, the four rotations and all eleven weight decoders probed
//! (layers 0 and 3 and the LM head) are bit-identical. The CPU model's and
//! the runner's 32-token continuations both equal the fork's text for all
//! three prompts of both bands.

#[path = "bonsai2_real/harness.rs"]
mod harness;

/// Capability-record name of G11.
const G11_TEST: &str =
    "oxibonsai-model::hybrid_metal_gates::hybrid_real_27b_metal_matches_cpu_tokens_bonsai2";
/// Capability-record name of G10.
const G10_TEST: &str =
    "oxibonsai-model::hybrid_metal_gates::hybrid_real_27b_metal_decode_throughput_bonsai2";

/// Serialises the two gates of this binary (one 27B mapping at a time).
static GATE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Holds [`GATE_LOCK`] and keeps `OXIBONSAI_KERNEL_TIER` cleared for one
/// gate, restoring its prior value on drop (also while unwinding). The CPU
/// model's `PQ2_0` GEMV honours that INT8 selector on any tier (K-14) while
/// the Metal runner never reads it, so a developer's exported value would
/// otherwise flip the CPU-vs-Metal comparison.
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

/// G11 on the real 27B: per-kernel parity on real activations, then greedy
/// runner tokens == CPU tokens == the fork's text, both bands.
#[test]
fn hybrid_real_27b_metal_matches_cpu_tokens_bonsai2() {
    let _env = TierEnvGuard::cleared();
    #[cfg(all(feature = "metal", target_os = "macos"))]
    gates::g11(G11_TEST);
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    skip_without_metal(G11_TEST);
}

/// G10 on the real 27B: Metal decode throughput, both bands.
#[test]
fn hybrid_real_27b_metal_decode_throughput_bonsai2() {
    let _env = TierEnvGuard::cleared();
    #[cfg(all(feature = "metal", target_os = "macos"))]
    gates::g10(G10_TEST);
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    skip_without_metal(G10_TEST);
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
    use oxibonsai_kernels::gated_delta_net::{GdnDims, GdnHeadOrder, GdnPath};
    use oxibonsai_kernels::gated_delta_net_chunk::gdn_prefill_with;
    use oxibonsai_kernels::gpu_backend::metal_full_layer::qwen35::{
        Qwen35LayerTrace, Qwen35MatrixId,
    };
    use oxibonsai_kernels::norms::{l2_norm_simd, rms_norm_gated_simd, sigmoid_mul_simd};
    use oxibonsai_kernels::rope_mrope::rope_partial_splithalf_simd;
    use oxibonsai_kernels::ssm_ops::causal_conv1d_k4_decode;
    use oxibonsai_kernels::{cpu_kernel_tier, silu_simd, swiglu_simd, KernelDispatcher};
    use oxibonsai_model::hybrid::metal::HybridMetalRunner;
    use oxibonsai_model::hybrid::model::HybridModel;
    use oxibonsai_model::hybrid::{FullAttnBlock, HybridBlock, LinearAttnBlock, RopeTables};
    use oxibonsai_model::layers::linear::LinearLayer;

    use crate::harness::{
        argmax, detokenize, gguf_vocab, golden_dir, gpt2_byte_decoder, log_softmax,
        parse_prompt_tokens, read_golden, read_golden_bytes, record_capability,
        require_model_files, show_bytes, top_n, PQ2_ENV, PROMPTS, PTQ1_ENV, REQUIRE_ENV, TOP_N,
    };

    /// KV window: the longest golden prompt (13 tokens) plus 32 generated
    /// tokens fit many times over; 256 MiB of `f16` KV on the GPU.
    const MAX_SEQ: usize = 4096;
    /// Tokens each golden continuation holds (`llama-cli -n 32`).
    const TEXT_STEPS: usize = 32;
    /// `llama-cli` appends this to every continuation dump.
    const TEXT_TRAILER: &[u8] = b"\n\n";
    /// Tokens timed per G10 run.
    const G10_TOKENS: usize = 32;
    /// G10's bar: decode tokens per second on the 24 GB M3.
    const G10_MIN_TOK_S: f64 = 5.0;
    /// Per-kernel parity bar on real activations (design §8.2).
    const KERNEL_COSINE: f64 = 0.999;
    /// Opt-in for the CPU throughput comparison printed beside G10.
    const CPU_THROUGHPUT_ENV: &str = "OXI_BONSAI2_CPU_THROUGHPUT";

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

    // ─────────────────────────────────────────────────────────────────────
    //  Locating the files (environment variables only)
    // ─────────────────────────────────────────────────────────────────────

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
    fn record(test: &str, ran: usize) {
        if ran == BANDS.len() {
            record_capability(true, test);
        } else {
            eprintln!("skip {test}: {ran} of {} bands located", BANDS.len());
            record_capability(false, test);
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
        let tier = cpu_kernel_tier();
        let kernel = Arc::new(KernelDispatcher::with_tier(tier));
        let t0 = Instant::now();
        let model = HybridModel::from_gguf_with(gguf, config, MAX_SEQ, &kernel)
            .unwrap_or_else(|e| panic!("{label}: the CPU model loads: {e}"));
        eprintln!(
            "{label}: CPU reference loaded in {:.2}s on kernel tier {tier} ({})",
            t0.elapsed().as_secs_f64(),
            model.describe()
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
        eprintln!(
            "{label}: Metal runner built in {:.2}s ({:.2} GB of weights in place, {:.0} MiB KV)",
            t0.elapsed().as_secs_f64(),
            runner.weight_bytes() as f64 / 1e9,
            runner.kv_cache_bytes() as f64 / (1024.0 * 1024.0)
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

    // ─────────────────────────────────────────────────────────────────────
    //  G11
    // ─────────────────────────────────────────────────────────────────────

    pub(super) fn g11(test: &str) {
        let bands = locate_bands(test);
        if bands.is_empty() {
            record(test, 0);
            return;
        }
        let mut failures = Vec::new();
        for (band, path) in &bands {
            failures.extend(g11_band(band, path));
        }
        assert!(
            failures.is_empty(),
            "G11 on the real 27B:\n{}",
            failures.join("\n")
        );
        record(test, bands.len());
    }

    fn g11_band(band: &Band, path: &Path) -> Vec<String> {
        let label = band.label;
        let mmap = mmap_gguf_file(path).unwrap_or_else(|e| panic!("mmap {}: {e}", path.display()));
        let gguf = GgufFile::parse(&mmap).expect("27B GGUF parses");
        let mut cpu = load_cpu(&gguf, label);
        let mut gpu = load_runner(&cpu, &mmap, label);
        let vocab_strings = gguf_vocab(&gguf);
        let decoder = gpt2_byte_decoder();

        let mut failures = kernel_parity(label, &cpu, &mut gpu, &golden_prompt(band, 1));

        let vocab = cpu.config().base.vocab_size;
        let (mut cl, mut gl) = (vec![0.0f32; vocab], vec![0.0f32; vocab]);
        for prompt_index in 1..=3usize {
            let tokens = golden_prompt(band, prompt_index);
            assert_eq!(
                detokenize(&vocab_strings, &decoder, &tokens),
                PROMPTS[prompt_index - 1].as_bytes(),
                "{label} prompt {prompt_index}: the golden tokens decode to the raw prompt"
            );
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
                "G11 {label} prompt {prompt_index}: prefill {} tokens: CPU {cpu_prefill:.2}s, \
                 Metal {gpu_prefill:.3}s",
                tokens.len()
            );

            let mut cpu_tokens = Vec::with_capacity(TEXT_STEPS);
            let mut gpu_tokens = Vec::with_capacity(TEXT_STEPS);
            let mut worst = (0.0f64, 0usize);
            for step in 0..TEXT_STEPS {
                let c = u32::try_from(argmax(&cl)).expect("token id");
                let g = u32::try_from(argmax(&gl)).expect("token id");
                let (dlp, gap) = compare_rows(&cl, &gl);
                if dlp > worst.0 {
                    worst = (dlp, step);
                }
                eprintln!(
                    "G11 {label} p{prompt_index} step {step:>2}: tokens cpu {c:>6} metal {g:>6} \
                     {} | worst top-{TOP_N} |dlogprob| {dlp:.3e} | cpu top-1/top-2 gap {gap:.4}",
                    if c == g { "ok " } else { "DIFF" }
                );
                if c != g {
                    failures.push(format!(
                        "{label} prompt {prompt_index} step {step}: Metal greedy token {g} != CPU \
                         {c} (CPU top-1/top-2 logprob gap {gap:.4})"
                    ));
                }
                cpu_tokens.push(c);
                gpu_tokens.push(g);
                if step + 1 < TEXT_STEPS {
                    // Both are fed the CPU's token, so a divergence is
                    // reported once and every later step stays comparable.
                    let pos = tokens.len() + step;
                    cpu.forward(c, pos, &mut cl)
                        .unwrap_or_else(|e| panic!("{label}: CPU decode at {pos}: {e}"));
                    gpu.forward_into(c, pos, &mut gl)
                        .unwrap_or_else(|e| panic!("{label}: Metal decode at {pos}: {e}"));
                }
            }
            let equal = cpu_tokens
                .iter()
                .zip(&gpu_tokens)
                .filter(|(a, b)| a == b)
                .count();
            let expected = golden_text(band, prompt_index);
            let cpu_text = detokenize(&vocab_strings, &decoder, &cpu_tokens);
            let gpu_text = detokenize(&vocab_strings, &decoder, &gpu_tokens);
            eprintln!(
                "G11 {label} prompt {prompt_index}: {equal}/{TEXT_STEPS} greedy tokens equal \
                 (Metal vs CPU); worst top-{TOP_N} |dlogprob| {:.3e} at step {}; text \
                 CPU {} Metal {}: \"{}\"",
                worst.0,
                worst.1,
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

    // ─────────────────────────────────────────────────────────────────────
    //  Per-kernel parity on real activations
    // ─────────────────────────────────────────────────────────────────────

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b) {
            dot += f64::from(x) * f64::from(y);
            na += f64::from(x) * f64::from(x);
            nb += f64::from(y) * f64::from(y);
        }
        if na == 0.0 && nb == 0.0 {
            1.0
        } else {
            dot / (na.sqrt() * nb.sqrt()).max(f64::MIN_POSITIVE)
        }
    }

    fn worst_rel(a: &[f32], b: &[f32]) -> f64 {
        let scale = b.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-30);
        f64::from(
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
                / scale,
        )
    }

    /// Collects one table row per checked stage and the failures.
    struct Report<'l> {
        label: &'l str,
        failures: Vec<String>,
    }

    impl Report<'_> {
        /// A float kernel: cosine at least [`KERNEL_COSINE`].
        fn close(&mut self, layer: usize, stage: &str, gpu: &[f32], cpu: &[f32]) {
            assert_eq!(gpu.len(), cpu.len(), "{stage}: lengths");
            let cos = cosine(gpu, cpu);
            let rel = worst_rel(gpu, cpu);
            let ok = cos >= KERNEL_COSINE && cos.is_finite();
            eprintln!(
                "G11 {} kernel parity layer {layer} {stage:<20} cosine {cos:.9} worst rel {rel:.3e} {}",
                self.label,
                if ok { "ok" } else { "FAIL" }
            );
            if !ok {
                self.failures.push(format!(
                    "{} layer {layer} {stage}: cosine {cos} < {KERNEL_COSINE}",
                    self.label
                ));
            }
        }

        /// A data-movement / exact-arithmetic stage: bit for bit.
        fn bitwise(&mut self, layer: usize, stage: &str, gpu: &[f32], cpu: &[f32]) {
            let diff = gpu
                .iter()
                .zip(cpu)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let ok = diff == 0 && gpu.len() == cpu.len();
            eprintln!(
                "G11 {} kernel parity layer {layer} {stage:<20} bitwise {} ({diff} of {} differ)",
                self.label,
                if ok { "ok" } else { "FAIL" },
                cpu.len()
            );
            if !ok {
                self.failures.push(format!(
                    "{} layer {layer} {stage}: {diff} of {} values differ from the CPU bit pattern",
                    self.label,
                    cpu.len()
                ));
            }
        }
    }

    fn stage<'t>(trace: &'t Qwen35LayerTrace, name: &str) -> &'t [f32] {
        trace
            .stage(name)
            .unwrap_or_else(|| panic!("the trace captures {name}"))
    }

    /// `f` applied to each `width`-wide row of `input`, into `out_width`-wide
    /// rows.
    fn per_row(
        input: &[f32],
        width: usize,
        out_width: usize,
        mut f: impl FnMut(&[f32], &mut [f32]),
    ) -> Vec<f32> {
        let rows = input.len() / width;
        let mut out = vec![0.0f32; rows * out_width];
        for (src, dst) in input.chunks(width).zip(out.chunks_mut(out_width)) {
            f(src, dst);
        }
        out
    }

    fn project(layer: &LinearLayer<'_>, input: &[f32]) -> Vec<f32> {
        per_row(input, layer.in_features(), layer.out_features(), |x, y| {
            layer.forward_vec(x, y).expect("CPU projection");
        })
    }

    fn rotate_rows(model: &HybridModel<'_>, x: &[f32], width: usize) -> Vec<f32> {
        let mut out = x.to_vec();
        if let Some(hook) = model.hadamard() {
            hook.rotate_rows_in_place(&mut out, width, x.len() / width)
                .expect("CPU rotation");
        }
        out
    }

    fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b).map(|(x, y)| x + y).collect()
    }

    /// Every stage of layer 0 (Gated DeltaNet) and layer 3 (full attention)
    /// against the CPU on real activations, plus the weight decoders.
    fn kernel_parity(
        label: &str,
        cpu: &HybridModel<'_>,
        gpu: &mut HybridMetalRunner<'_>,
        tokens: &[u32],
    ) -> Vec<String> {
        let mut report = Report {
            label,
            failures: Vec::new(),
        };
        let hidden = cpu.config().base.hidden_size;

        // Embedding lookup + inverse rotation: bit-identical by construction.
        let rows0 = gpu.embed(tokens).expect("embed");
        let mut cpu_rows = vec![0.0f32; tokens.len() * hidden];
        for (t, &token) in tokens.iter().enumerate() {
            let row = &mut cpu_rows[t * hidden..(t + 1) * hidden];
            cpu.embedding()
                .row(token, hidden, row)
                .expect("CPU embedding");
            if let Some(hook) = cpu.hadamard() {
                hook.inverse_embedding(row).expect("CPU inverse rotation");
            }
        }
        report.bitwise(0, "embedding", &rows0, &cpu_rows);

        let first_linear = cpu
            .blocks()
            .iter()
            .position(|b| !b.is_full())
            .expect("a Gated-DeltaNet layer");
        let first_full = cpu
            .blocks()
            .iter()
            .position(HybridBlock::is_full)
            .expect("a full-attention layer");

        // The input of the first full layer: the GPU's own residual stream.
        gpu.reset();
        let (dump, _) = gpu.forward_with_dump(tokens, 0).expect("Metal dump");
        let rows_full = if first_full == 0 {
            rows0.clone()
        } else {
            dump[first_full - 1].clone()
        };
        let rows_linear = if first_linear == 0 {
            rows0.clone()
        } else {
            dump[first_linear - 1].clone()
        };

        gpu.reset();
        let trace = gpu
            .trace_layer(first_linear, &rows_linear, 0)
            .expect("trace the Gated-DeltaNet layer");
        if let Some(HybridBlock::Linear(block)) = cpu.block(first_linear) {
            linear_parity(&mut report, cpu, block, &trace, &rows_linear);
        }
        gpu.reset();
        let trace = gpu
            .trace_layer(first_full, &rows_full, 0)
            .expect("trace the full-attention layer");
        if let Some(HybridBlock::Full(block)) = cpu.block(first_full) {
            full_parity(&mut report, cpu, block, &trace, &rows_full, tokens.len());
        }

        decoder_parity(&mut report, cpu, gpu, first_linear, first_full);
        gpu.reset();
        report.failures
    }

    /// The FFN half, shared by both layer kinds.
    fn ffn_parity(
        report: &mut Report<'_>,
        cpu: &HybridModel<'_>,
        layer: usize,
        norm: &oxibonsai_model::layers::rms_norm::RmsNorm,
        mats: [&LinearLayer<'_>; 3],
        trace: &Qwen35LayerTrace,
    ) {
        let c = cpu.config();
        let (hidden, inter) = (c.base.hidden_size, c.base.intermediate_size);
        let resid = stage(trace, "attn_residual");
        let normed = per_row(resid, hidden, hidden, |x, y| {
            norm.forward(x, y).expect("CPU norm");
        });
        report.close(layer, "ffn_norm", stage(trace, "ffn_norm"), &normed);
        report.bitwise(
            layer,
            "ffn_norm_rotated",
            stage(trace, "ffn_norm_rotated"),
            &rotate_rows(cpu, stage(trace, "ffn_norm"), hidden),
        );
        let x = stage(trace, "ffn_norm_rotated");
        report.close(
            layer,
            "ffn_gate",
            stage(trace, "ffn_gate"),
            &project(mats[0], x),
        );
        report.close(
            layer,
            "ffn_up",
            stage(trace, "ffn_up"),
            &project(mats[1], x),
        );
        let mut act = vec![0.0f32; stage(trace, "ffn_gate").len()];
        swiglu_simd(stage(trace, "ffn_gate"), stage(trace, "ffn_up"), &mut act)
            .expect("CPU swiglu");
        report.close(
            layer,
            "ffn_act_rotated",
            stage(trace, "ffn_act_rotated"),
            &rotate_rows(cpu, &act, inter),
        );
        let down = project(mats[2], stage(trace, "ffn_act_rotated"));
        report.close(
            layer,
            "residual",
            stage(trace, "residual"),
            &add(resid, &down),
        );
    }

    fn linear_parity(
        report: &mut Report<'_>,
        cpu: &HybridModel<'_>,
        block: &LinearAttnBlock<'_>,
        trace: &Qwen35LayerTrace,
        rows: &[f32],
    ) {
        let layer = block.layer_idx();
        let c = cpu.config();
        let hidden = c.base.hidden_size;
        let (nk, nv, hk, hv) = (c.n_k_heads(), c.n_v_heads(), c.head_k_dim(), c.head_v_dim());
        let conv_dim = c.conv_dim();
        let inner = nv * hv;
        let eps = c.base.rms_norm_eps;
        let t_len = rows.len() / hidden;
        let map = cpu.vhead_map();

        let normed = per_row(rows, hidden, hidden, |x, y| {
            block.attn_norm().forward(x, y).expect("CPU norm");
        });
        report.close(layer, "attn_norm", stage(trace, "attn_norm"), &normed);
        report.bitwise(
            layer,
            "attn_norm_rotated",
            stage(trace, "attn_norm_rotated"),
            &rotate_rows(cpu, stage(trace, "attn_norm"), hidden),
        );
        let x = stage(trace, "attn_norm_rotated");
        report.close(
            layer,
            "attn_qkv",
            stage(trace, "attn_qkv"),
            &project(block.attn_qkv(), x),
        );
        report.close(
            layer,
            "attn_gate",
            stage(trace, "attn_gate"),
            &project(block.attn_gate(), x),
        );
        let ab = per_row(stage(trace, "attn_norm"), hidden, 2 * nv, |x, y| {
            let (alpha, beta) = y.split_at_mut(nv);
            block.ssm_alpha().forward_vec(x, alpha).expect("CPU alpha");
            block.ssm_beta().forward_vec(x, beta).expect("CPU beta");
        });
        report.close(layer, "ssm_alpha_beta", stage(trace, "ssm_alpha_beta"), &ab);

        // Conv + SiLU from the GPU's projection, fresh window.
        let qkv = stage(trace, "attn_qkv");
        let mut window = vec![0.0f32; conv_dim * (c.ssm_conv_kernel - 1)];
        let mut conv = vec![0.0f32; conv_dim];
        let mut conv_silu = vec![0.0f32; t_len * conv_dim];
        for t in 0..t_len {
            causal_conv1d_k4_decode(
                &mut window,
                &qkv[t * conv_dim..(t + 1) * conv_dim],
                block.ssm_conv1d(),
                &mut conv,
            )
            .expect("CPU conv");
            silu_simd(&conv, &mut conv_silu[t * conv_dim..(t + 1) * conv_dim]).expect("CPU silu");
        }
        report.close(layer, "conv_silu", stage(trace, "conv_silu"), &conv_silu);

        // L2 norm + gated delta rule from the GPU's conv output and gates.
        let conv_out = stage(trace, "conv_silu");
        let gpu_ab = stage(trace, "ssm_alpha_beta");
        let mut q = vec![0.0f32; t_len * nk * hk];
        let mut k = vec![0.0f32; t_len * nk * hk];
        let mut v = vec![0.0f32; t_len * inner];
        let mut alpha = vec![0.0f32; t_len * nv];
        let mut beta = vec![0.0f32; t_len * nv];
        for t in 0..t_len {
            let row = &conv_out[t * conv_dim..(t + 1) * conv_dim];
            for h in 0..nk {
                let dst = (t * nk + h) * hk;
                l2_norm_simd(&row[h * hk..(h + 1) * hk], &mut q[dst..dst + hk], eps)
                    .expect("CPU l2 q");
                let src = nk * hk + h * hk;
                l2_norm_simd(&row[src..src + hk], &mut k[dst..dst + hk], eps).expect("CPU l2 k");
            }
            map.gather_grouped(&row[2 * nk * hk..], hv, &mut v[t * inner..(t + 1) * inner])
                .expect("gather v");
            let ab_row = &gpu_ab[t * 2 * nv..(t + 1) * 2 * nv];
            map.gather_grouped_scalar(&ab_row[..nv], &mut alpha[t * nv..(t + 1) * nv])
                .expect("gather alpha");
            map.gather_grouped_scalar(&ab_row[nv..], &mut beta[t * nv..(t + 1) * nv])
                .expect("gather beta");
        }
        let dims = GdnDims::new(nk, nv, hk, hv);
        let gates = block.gates().gates(&alpha, &beta);
        let mut state = vec![0.0f32; nv * hv * hk];
        let mut gdn = vec![0.0f32; t_len * inner];
        gdn_prefill_with(
            &mut state,
            &q,
            &k,
            &v,
            &gates,
            &mut gdn,
            t_len,
            &dims,
            GdnHeadOrder::Grouped,
            GdnPath::Fused,
        )
        .expect("CPU gated delta rule");
        report.close(layer, "gdn", stage(trace, "gdn"), &gdn);

        // Gated RMSNorm per grouped v-head (z regrouped) + rotation.
        let gpu_gdn = stage(trace, "gdn");
        let z = stage(trace, "attn_gate");
        let mut z_grouped = vec![0.0f32; inner];
        let mut gated = vec![0.0f32; t_len * inner];
        for t in 0..t_len {
            map.gather_grouped(&z[t * inner..(t + 1) * inner], hv, &mut z_grouped)
                .expect("gather z");
            for m in 0..nv {
                let lo = t * inner + m * hv;
                rms_norm_gated_simd(
                    &gpu_gdn[lo..lo + hv],
                    block.ssm_norm().weight(),
                    &z_grouped[m * hv..(m + 1) * hv],
                    &mut gated[lo..lo + hv],
                    eps,
                )
                .expect("CPU gated norm");
            }
        }
        report.close(
            layer,
            "gated_norm_rotated",
            stage(trace, "gated_norm_rotated"),
            &rotate_rows(cpu, &gated, inner),
        );
        let out = project(block.ssm_out(), stage(trace, "gated_norm_rotated"));
        report.close(
            layer,
            "attn_residual",
            stage(trace, "attn_residual"),
            &add(rows, &out),
        );

        ffn_parity(
            report,
            cpu,
            layer,
            block.post_attn_norm(),
            [block.ffn_gate(), block.ffn_up(), block.ffn_down()],
            trace,
        );
    }

    fn full_parity(
        report: &mut Report<'_>,
        cpu: &HybridModel<'_>,
        block: &FullAttnBlock<'_>,
        trace: &Qwen35LayerTrace,
        rows: &[f32],
        t_len: usize,
    ) {
        let layer = block.layer_idx();
        let c = cpu.config();
        let hidden = c.base.hidden_size;
        let (nh, nkv, hd) = (
            c.base.num_attention_heads,
            c.base.num_kv_heads,
            c.base.head_dim,
        );
        let heads = nh * hd;
        let kv_width = nkv * hd;
        let n_rot = c.rope_dimension_count;

        let normed = per_row(rows, hidden, hidden, |x, y| {
            block.attn_norm().forward(x, y).expect("CPU norm");
        });
        report.close(layer, "attn_norm", stage(trace, "attn_norm"), &normed);
        report.bitwise(
            layer,
            "attn_norm_rotated",
            stage(trace, "attn_norm_rotated"),
            &rotate_rows(cpu, stage(trace, "attn_norm"), hidden),
        );
        let x = stage(trace, "attn_norm_rotated");
        report.close(
            layer,
            "attn_q",
            stage(trace, "attn_q"),
            &project(block.attn_q(), x),
        );
        report.close(
            layer,
            "attn_k",
            stage(trace, "attn_k"),
            &project(block.attn_k(), x),
        );
        report.close(
            layer,
            "attn_v",
            stage(trace, "attn_v"),
            &project(block.attn_v(), x),
        );

        // Per-head q/k RMSNorm + partial RoPE from the GPU's projections.
        let rope = RopeTables::new(n_rot, t_len.max(1), c.base.rope_freq_base).expect("rope");
        let q_all = stage(trace, "attn_q");
        let k_raw = stage(trace, "attn_k");
        let mut q_rope = vec![0.0f32; t_len * heads];
        let mut k_rope = vec![0.0f32; t_len * kv_width];
        let mut tmp = vec![0.0f32; hd];
        for t in 0..t_len {
            let (cos, sin) = rope.angles(t).expect("angles");
            for h in 0..nh {
                let src = t * 2 * heads + h * 2 * hd;
                block
                    .attn_q_norm()
                    .forward(&q_all[src..src + hd], &mut tmp)
                    .expect("CPU q norm");
                let dst = t * heads + h * hd;
                rope_partial_splithalf_simd(&tmp, &mut q_rope[dst..dst + hd], hd, n_rot, cos, sin)
                    .expect("CPU rope q");
            }
            for h in 0..nkv {
                let lo = t * kv_width + h * hd;
                block
                    .attn_k_norm()
                    .forward(&k_raw[lo..lo + hd], &mut tmp)
                    .expect("CPU k norm");
                rope_partial_splithalf_simd(&tmp, &mut k_rope[lo..lo + hd], hd, n_rot, cos, sin)
                    .expect("CPU rope k");
            }
        }
        report.close(layer, "q_rope", stage(trace, "q_rope"), &q_rope);
        report.close(layer, "k_rope", stage(trace, "k_rope"), &k_rope);

        // Causal GQA over the GPU's own keys/values rounded to f16 (the
        // cache's element type), in f64.
        let (gq, gk, gv) = (
            stage(trace, "q_rope"),
            stage(trace, "k_rope"),
            stage(trace, "attn_v"),
        );
        let widen = |x: f32| f64::from(half::f16::from_f32(x).to_f32());
        let scale = 1.0 / (hd as f64).sqrt();
        let mut attn = vec![0.0f32; t_len * heads];
        for t in 0..t_len {
            for h in 0..nh {
                let kv = h / (nh / nkv);
                let qv = &gq[t * heads + h * hd..t * heads + (h + 1) * hd];
                let scores: Vec<f64> = (0..=t)
                    .map(|p| {
                        let key = &gk[p * kv_width + kv * hd..p * kv_width + (kv + 1) * hd];
                        qv.iter()
                            .zip(key)
                            .map(|(&a, &b)| f64::from(a) * widen(b))
                            .sum::<f64>()
                            * scale
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let total: f64 = weights.iter().sum();
                for d in 0..hd {
                    let acc: f64 = weights
                        .iter()
                        .enumerate()
                        .map(|(p, w)| w * widen(gv[p * kv_width + kv * hd + d]))
                        .sum();
                    attn[t * heads + h * hd + d] = (acc / total) as f32;
                }
            }
        }
        report.close(layer, "attention", stage(trace, "attention"), &attn);

        // Sigmoid output gate (the gate half of attn_q) + rotation.
        let gpu_attn = stage(trace, "attention");
        let gate: Vec<f32> = (0..t_len * heads)
            .map(|i| {
                let (t, j) = (i / heads, i % heads);
                q_all[t * 2 * heads + (j / hd) * 2 * hd + hd + j % hd]
            })
            .collect();
        let mut gated = vec![0.0f32; t_len * heads];
        sigmoid_mul_simd(gpu_attn, &gate, &mut gated).expect("CPU sigmoid gate");
        report.close(
            layer,
            "attn_gated_rotated",
            stage(trace, "attn_gated_rotated"),
            &rotate_rows(cpu, &gated, heads),
        );
        let out = project(block.attn_output(), stage(trace, "attn_gated_rotated"));
        report.close(
            layer,
            "attn_residual",
            stage(trace, "attn_residual"),
            &add(rows, &out),
        );

        ffn_parity(
            report,
            cpu,
            layer,
            block.post_attn_norm(),
            [block.ffn_gate(), block.ffn_up(), block.ffn_down()],
            trace,
        );
    }

    /// Every weight decoder bit for bit: a one-hot GEMV returns one
    /// dequantized column, on the GPU and on the CPU.
    fn decoder_parity(
        report: &mut Report<'_>,
        cpu: &HybridModel<'_>,
        gpu: &mut HybridMetalRunner<'_>,
        first_linear: usize,
        first_full: usize,
    ) {
        let mut probe =
            |layer: usize, id: Qwen35MatrixId, name: &str, cpu_layer: &LinearLayer<'_>| {
                let cols = cpu_layer.in_features();
                let mut gpu_cols = Vec::new();
                let mut cpu_cols = Vec::new();
                // Stage-1, stage-2 and qh elements of a PTQ1_0 super-block,
                // the next block, the middle and the last column.
                for col in [0, 1, 80, 100, 120, 127, 128, cols / 2 + 5, cols - 1] {
                    let mut x = vec![0.0f32; cols];
                    x[col] = 1.0;
                    gpu_cols.extend(gpu.gemv_probe(layer, id, &x).expect("GPU one-hot probe"));
                    let mut y = vec![0.0f32; cpu_layer.out_features()];
                    cpu_layer
                        .forward_vec(&x, &mut y)
                        .expect("CPU one-hot probe");
                    cpu_cols.extend(y);
                }
                // Value equality: a zero code may come back as -0.0 on one side.
                let diff = gpu_cols
                    .iter()
                    .zip(&cpu_cols)
                    .filter(|(a, b)| a != b)
                    .count();
                let stage_name = format!("decode {name}");
                if diff == 0 {
                    eprintln!(
                    "G11 {} kernel parity layer {layer} {stage_name:<20} bitwise ok ({} values)",
                    report.label,
                    cpu_cols.len()
                );
                } else {
                    report.bitwise(layer, &stage_name, &gpu_cols, &cpu_cols);
                }
            };
        if let Some(HybridBlock::Linear(b)) = cpu.block(first_linear) {
            probe(
                first_linear,
                Qwen35MatrixId::AttnQkv,
                "attn_qkv",
                b.attn_qkv(),
            );
            probe(
                first_linear,
                Qwen35MatrixId::AttnGate,
                "attn_gate",
                b.attn_gate(),
            );
            probe(first_linear, Qwen35MatrixId::SsmOut, "ssm_out", b.ssm_out());
            probe(
                first_linear,
                Qwen35MatrixId::FfnGate,
                "ffn_gate",
                b.ffn_gate(),
            );
            probe(first_linear, Qwen35MatrixId::FfnUp, "ffn_up", b.ffn_up());
            probe(
                first_linear,
                Qwen35MatrixId::FfnDown,
                "ffn_down",
                b.ffn_down(),
            );
        }
        if let Some(HybridBlock::Full(b)) = cpu.block(first_full) {
            probe(first_full, Qwen35MatrixId::AttnQ, "attn_q", b.attn_q());
            probe(first_full, Qwen35MatrixId::AttnK, "attn_k", b.attn_k());
            probe(first_full, Qwen35MatrixId::AttnV, "attn_v", b.attn_v());
            probe(
                first_full,
                Qwen35MatrixId::AttnOutput,
                "attn_output",
                b.attn_output(),
            );
        }
        probe(
            cpu.blocks().len(),
            Qwen35MatrixId::LmHead,
            "output",
            cpu.lm_head(),
        );
    }

    // ─────────────────────────────────────────────────────────────────────
    //  G10
    // ─────────────────────────────────────────────────────────────────────

    pub(super) fn g10(test: &str) {
        let bands = locate_bands(test);
        if bands.is_empty() {
            record(test, 0);
            return;
        }
        let mut failures = Vec::new();
        for (band, path) in &bands {
            let best = g10_band(band, path);
            if best < G10_MIN_TOK_S {
                failures.push(format!(
                    "{}: best Metal decode {best:.2} tok/s < {} tok/s",
                    band.label, G10_MIN_TOK_S
                ));
            }
        }
        assert!(failures.is_empty(), "G10:\n{}", failures.join("\n"));
        record(test, bands.len());
    }

    /// Best-of-three Metal decode rate of one band, in tokens per second.
    fn g10_band(band: &Band, path: &Path) -> f64 {
        let label = band.label;
        let mmap = mmap_gguf_file(path).unwrap_or_else(|e| panic!("mmap {}: {e}", path.display()));
        let gguf = GgufFile::parse(&mmap).expect("27B GGUF parses");
        let mut cpu = load_cpu(&gguf, label);
        let mut gpu = load_runner(&cpu, &mmap, label);
        let prompt = golden_prompt(band, 1);
        let vocab = gpu.vocab_size();
        let mut logits = vec![0.0f32; vocab];

        // Warm-up: pages every weight in and builds every pipeline.
        gpu.forward_prefill(&prompt, 0, &mut logits)
            .expect("warm-up prefill");
        for step in 0..4 {
            let token = u32::try_from(argmax(&logits)).expect("token id");
            gpu.forward_into(token, prompt.len() + step, &mut logits)
                .expect("warm-up decode");
        }

        let mut best = 0.0f64;
        for run in 1..=3 {
            gpu.reset();
            let t0 = Instant::now();
            gpu.forward_prefill(&prompt, 0, &mut logits)
                .expect("prefill");
            let prefill = t0.elapsed().as_secs_f64();
            let mut gpu_seconds = 0.0f64;
            let t0 = Instant::now();
            for step in 0..G10_TOKENS {
                let token = u32::try_from(argmax(&logits)).expect("token id");
                gpu.forward_into(token, prompt.len() + step, &mut logits)
                    .expect("decode");
                gpu_seconds += gpu.last_gpu_seconds();
            }
            let elapsed = t0.elapsed().as_secs_f64();
            let rate = G10_TOKENS as f64 / elapsed.max(1e-9);
            best = best.max(rate);
            eprintln!(
                "G10 {label} run {run}: {G10_TOKENS} tokens in {elapsed:.3}s = {rate:.2} tok/s \
                 (GPU {:.1} ms/token of {:.1} ms wall; prefill {} tokens {prefill:.3}s); load \
                 average {}",
                gpu_seconds * 1e3 / G10_TOKENS as f64,
                elapsed * 1e3 / G10_TOKENS as f64,
                prompt.len(),
                load_average()
            );
        }
        eprintln!(
            "G10 {label}: best Metal decode {best:.2} tok/s (bar {} tok/s)",
            G10_MIN_TOK_S
        );

        if std::env::var(CPU_THROUGHPUT_ENV).is_ok_and(|v| v.trim() == "1") {
            cpu_throughput(label, &mut cpu, &prompt);
        }
        best
    }

    /// The CPU forward's decode rate (best of three 8-token runs), printed
    /// for comparison; not a gate.
    fn cpu_throughput(label: &str, cpu: &mut HybridModel<'_>, prompt: &[u32]) {
        const CPU_TOKENS: usize = 8;
        let vocab = cpu.config().base.vocab_size;
        let mut logits = vec![0.0f32; vocab];
        let mut best = 0.0f64;
        for run in 1..=3 {
            cpu.reset();
            cpu.forward_prefill(prompt, 0, &mut logits)
                .expect("CPU prefill");
            let t0 = Instant::now();
            for step in 0..CPU_TOKENS {
                let token = u32::try_from(argmax(&logits)).expect("token id");
                cpu.forward(token, prompt.len() + step, &mut logits)
                    .expect("CPU decode");
            }
            let elapsed = t0.elapsed().as_secs_f64();
            let rate = CPU_TOKENS as f64 / elapsed.max(1e-9);
            best = best.max(rate);
            eprintln!(
                "G10 {label} CPU run {run}: {CPU_TOKENS} tokens in {elapsed:.2}s = {rate:.3} tok/s; \
                 load average {}",
                load_average()
            );
        }
        eprintln!("G10 {label}: best CPU decode {best:.3} tok/s (informational)");
        cpu_op_shares(label, cpu, prompt.len() + CPU_TOKENS, &mut logits);
    }

    /// One timing of `f`, in seconds.
    fn time_once(f: impl FnOnce()) -> f64 {
        let t0 = Instant::now();
        f();
        t0.elapsed().as_secs_f64()
    }

    fn ramp(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i as f32) * 0.37).sin()).collect()
    }

    /// Where one CPU decode step's time goes, for a CPU-performance pass to
    /// start from. Five rounds (`OP_ROUNDS`), each one decode step (timed)
    /// immediately followed by every op family of a step run on its own at
    /// the real shapes and the step's context — the step's projections
    /// (quantized GEMV), the two BF16 gate projections, the Hadamard
    /// rotations, the Gated-DeltaNet block (conv, SiLU, L2 norms,
    /// recurrence, gated norm) and the full-attention core (q/k norms,
    /// partial RoPE, attention over the cached history); the rest
    /// (embedding, RMSNorms, SwiGLU, sigmoid gate, residual adds, v-head
    /// gathers, copies) is the remainder. Each figure is its minimum over
    /// the rounds — contention from other load only ever inflates a timing,
    /// so the minimum is the least-disturbed sample of each — and families
    /// whose minima still add up to more than the fastest step are flagged.
    fn cpu_op_shares(label: &str, cpu: &mut HybridModel<'_>, mut pos: usize, logits: &mut [f32]) {
        const OP_ROUNDS: usize = 5;
        let mut step_seconds = f64::INFINITY;
        let mut families = [f64::INFINITY; 5];
        for _ in 0..OP_ROUNDS {
            let token = u32::try_from(argmax(logits)).expect("token id");
            let step = time_once(|| {
                cpu.forward(token, pos, logits).expect("CPU decode");
            });
            pos += 1;
            step_seconds = step_seconds.min(step);
            for (best, sample) in families.iter_mut().zip(time_families(cpu, pos)) {
                *best = best.min(sample);
            }
        }
        let context = pos;
        let [gemv, gates, fwht, gdn, attention] = families;
        let measured = gemv + gates + fwht + gdn + attention;
        let rest = (step_seconds - measured).max(0.0);
        let share = |s: f64| 100.0 * s / step_seconds.max(1e-12);
        eprintln!(
            "O3 {label}: CPU decode step {:.1} ms (fastest of {OP_ROUNDS}) at context ≤ {context}; \
             per-op, each family run alone right after a step, fastest of {OP_ROUNDS}; load \
             average {}",
            step_seconds * 1e3,
            load_average()
        );
        for (name, s) in [
            ("GEMV (quantized projections + LM head)", gemv),
            ("BF16 gate projections (ssm_alpha/beta)", gates),
            ("FWHT (Hadamard rotations)", fwht),
            ("GDN (conv, SiLU, L2, recurrence, gated norm)", gdn),
            ("attention (q/k norm, RoPE, attend)", attention),
            ("other (remainder of the step)", rest),
        ] {
            eprintln!(
                "O3 {label}:   {name:<46} {:9.1} ms  {:5.1}%",
                s * 1e3,
                share(s)
            );
        }
        if measured > step_seconds {
            eprintln!(
                "O3 {label}: WARNING the families add up to {:.1} ms, more than the {:.1} ms step \
                 — the machine was contended while they ran; do not read these shares as a \
                 profile",
                measured * 1e3,
                step_seconds * 1e3
            );
        }
    }

    /// One timing of each op family of a decode step (see
    /// [`cpu_op_shares`]), in seconds: `[gemv, bf16 gates, fwht, gdn,
    /// attention]`.
    fn time_families(cpu: &HybridModel<'_>, context: usize) -> [f64; 5] {
        let c = cpu.config();
        let hidden = c.base.hidden_size;
        let (nk, nv, hk, hv) = (c.n_k_heads(), c.n_v_heads(), c.head_k_dim(), c.head_v_dim());
        let (nh, nkv, hd) = (
            c.base.num_attention_heads,
            c.base.num_kv_heads,
            c.base.head_dim,
        );
        let eps = c.base.rms_norm_eps;

        // Quantized projections: every matrix of every layer, plus the head.
        let mut mats: Vec<&LinearLayer<'_>> = Vec::new();
        for block in cpu.blocks() {
            match block {
                HybridBlock::Full(b) => mats.extend([
                    b.attn_q(),
                    b.attn_k(),
                    b.attn_v(),
                    b.attn_output(),
                    b.ffn_gate(),
                    b.ffn_up(),
                    b.ffn_down(),
                ]),
                HybridBlock::Linear(b) => mats.extend([
                    b.attn_qkv(),
                    b.attn_gate(),
                    b.ssm_out(),
                    b.ffn_gate(),
                    b.ffn_up(),
                    b.ffn_down(),
                ]),
            }
        }
        mats.push(cpu.lm_head());
        let xs: Vec<Vec<f32>> = mats.iter().map(|m| ramp(m.in_features())).collect();
        let mut ys: Vec<Vec<f32>> = mats.iter().map(|m| vec![0.0; m.out_features()]).collect();
        let gemv = time_once(|| {
            for ((m, x), y) in mats.iter().zip(&xs).zip(ys.iter_mut()) {
                m.forward_vec(x, y).expect("CPU projection");
            }
        });

        // The un-folded BF16 gate projections of the linear layers.
        let linear: Vec<&LinearAttnBlock<'_>> = cpu
            .blocks()
            .iter()
            .filter_map(HybridBlock::as_linear)
            .collect();
        let full: Vec<&FullAttnBlock<'_>> = cpu
            .blocks()
            .iter()
            .filter_map(HybridBlock::as_full)
            .collect();
        let x_hidden = ramp(hidden);
        let mut gate_out = vec![0.0f32; nv];
        let gates = time_once(|| {
            for b in &linear {
                b.ssm_alpha()
                    .forward_vec(&x_hidden, &mut gate_out)
                    .expect("alpha");
                b.ssm_beta()
                    .forward_vec(&x_hidden, &mut gate_out)
                    .expect("beta");
            }
        });

        // Hadamard rotations: four activations per layer, the LM head's
        // input and the embedding row's inverse.
        let heads_width = nh * hd;
        let inner = nv * hv;
        let mut rot_bufs: Vec<(Vec<f32>, usize)> = Vec::new();
        for block in cpu.blocks() {
            let second = if block.is_full() { heads_width } else { inner };
            for width in [hidden, second, hidden, c.base.intermediate_size] {
                rot_bufs.push((ramp(width), width));
            }
        }
        rot_bufs.push((ramp(hidden), hidden));
        let mut embed_row = ramp(hidden);
        let fwht = match cpu.hadamard() {
            Some(hook) => time_once(|| {
                for (buf, width) in rot_bufs.iter_mut() {
                    hook.rotate_rows_in_place(buf, *width, 1).expect("rotate");
                }
                hook.inverse_embedding(&mut embed_row)
                    .expect("inverse rotation");
            }),
            None => 0.0,
        };

        // Gated-DeltaNet block of every linear layer.
        let conv_dim = c.conv_dim();
        let mut window = vec![0.0f32; conv_dim * (c.ssm_conv_kernel - 1)];
        let conv_in = ramp(conv_dim);
        let mut conv = vec![0.0f32; conv_dim];
        let mut act = vec![0.0f32; conv_dim];
        let (mut q, mut k) = (vec![0.0f32; nk * hk], vec![0.0f32; nk * hk]);
        let v = ramp(inner);
        let (alpha, beta) = (ramp(nv), ramp(nv));
        let mut state = vec![0.0f32; nv * hv * hk];
        let mut out = vec![0.0f32; inner];
        let z = ramp(inner);
        let mut gated = vec![0.0f32; inner];
        let dims = GdnDims::new(nk, nv, hk, hv);
        let gdn = time_once(|| {
            for b in &linear {
                causal_conv1d_k4_decode(&mut window, &conv_in, b.ssm_conv1d(), &mut conv)
                    .expect("conv");
                silu_simd(&conv, &mut act).expect("silu");
                for h in 0..nk {
                    l2_norm_simd(
                        &act[h * hk..(h + 1) * hk],
                        &mut q[h * hk..(h + 1) * hk],
                        eps,
                    )
                    .expect("l2 q");
                    let lo = nk * hk + h * hk;
                    l2_norm_simd(&act[lo..lo + hk], &mut k[h * hk..(h + 1) * hk], eps)
                        .expect("l2 k");
                }
                let g = b.gates().gates(&alpha, &beta);
                gdn_prefill_with(
                    &mut state,
                    &q,
                    &k,
                    &v,
                    &g,
                    &mut out,
                    1,
                    &dims,
                    GdnHeadOrder::Grouped,
                    GdnPath::Fused,
                )
                .expect("recurrence");
                for m in 0..nv {
                    rms_norm_gated_simd(
                        &out[m * hv..(m + 1) * hv],
                        b.ssm_norm().weight(),
                        &z[m * hv..(m + 1) * hv],
                        &mut gated[m * hv..(m + 1) * hv],
                        eps,
                    )
                    .expect("gated norm");
                }
            }
        });

        // Full-attention core: q/k norms + partial RoPE + attention over the
        // cached history.
        let rope = RopeTables::new(c.rope_dimension_count, context + 1, c.base.rope_freq_base)
            .expect("rope");
        let (cos, sin) = rope.angles(context).expect("angles");
        let head_in = ramp(hd);
        let mut tmp = vec![0.0f32; hd];
        let mut roped = vec![0.0f32; hd];
        let group = nh / nkv;
        let queries = ramp(group * hd);
        let mut attn_out = vec![0.0f32; group * hd];
        let attention = time_once(|| {
            for b in &full {
                for h in 0..nh + nkv {
                    let norm = if h < nh {
                        b.attn_q_norm()
                    } else {
                        b.attn_k_norm()
                    };
                    norm.forward(&head_in, &mut tmp).expect("q/k norm");
                    rope_partial_splithalf_simd(
                        &tmp,
                        &mut roped,
                        hd,
                        c.rope_dimension_count,
                        cos,
                        sin,
                    )
                    .expect("rope");
                }
                for kh in 0..nkv {
                    cpu.kv_cache()
                        .attend_group(b.kv_slot(), kh, context, &queries, &mut attn_out)
                        .expect("attend");
                }
            }
        });

        [gemv, gates, fwht, gdn, attention]
    }
}
