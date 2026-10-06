//! Real-model vision gate on Metal (bonsai2-design.md §6.2 / §8.2): the
//! Bonsai 2 27B on the Metal hybrid runner with the Metal vision tower, both
//! bands (`PQ2_0`, `PTQ1_0`), against the CPU path and the PrismML fork's
//! vision goldens.
//!
//! # The two paths
//!
//! * **CPU** — the CPU tower's rows through the CPU model's rows prefill
//!   (`HybridModel::forward_prefill_pieces`), then greedy decoding on the
//!   CPU model: the path the landed CPU gate pins to the fork's `.cpu`
//!   golden (32/32 and 16/16 ids).
//! * **Metal** — the Metal tower's rows (the projector's `Q8_0` / `F16`
//!   weights read exactly on the GPU), assembled by the CPU model at the
//!   runner's rope start (`HybridModel::assemble_prompt_at`) and prefilled
//!   by the runner (`HybridMetalRunner::forward_prefill_rows`, the batched
//!   GEMM prefill), then decoded on the runner — teacher-forced on the CPU
//!   path's ids so both see the same context at every step.
//!
//! # What is checked, per band and prompt
//!
//! 1. The prompts, rendered through the GGUF's own chat template with an
//!    image part and a text part, splice to one bracketed `<|image_pad|>`
//!    of 48 image rows: 67 / 75 prompt rows, the fork server's own
//!    `usage.prompt_tokens`. Both towers encode the fixture to the same
//!    6 x 8 merged grid.
//! 2. **Metal == CPU**: the runner's greedy token equals the CPU path's at
//!    every generated step, and the two paths' log-probabilities agree to
//!    the G4 logprob band (2e-2) over the CPU top-10 and the golden
//!    top-10 at every step.
//! 3. **Against the golden** (the fork's `.cpu` ids; `.metal` has the same
//!    ids): the first 16 steps exact on both paths (the landed
//!    CPU gate's floor), every step's top-1 equal to the golden's while the
//!    context is the golden's, and any divergence reported with the golden
//!    top-2 gap and both paths' `|Δlogprob|` at that step. The `PTQ1_0`
//!    band is held to the same `PQ2_0` golden (the two files encode the
//!    same ternary weights; the text gates use the same oracle).
//! 4. **A ceiling against both goldens**: four distances per prompt — the
//!    Metal path to the fork's `.metal` golden, the CPU path to `.metal`,
//!    the Metal path to `.cpu` and the CPU path to `.cpu` — each the worst
//!    `|Δlogprob|` over the golden step's top-10, taken over the steps
//!    whose context is still the golden's, are all printed and must each
//!    stay within `GOLDEN_CEILING` (1.25e-1). The 2e-2 band of point 2
//!    cannot hold against the fork: the fork's own two backends differ by
//!    6.8e-2 / 6.4e-2 on these prompts and the CPU path sits 4.7e-2 /
//!    6.2e-2 from the `.metal` golden. So the band is enforced between our
//!    two paths, and the ceiling bounds how far either path may drift from
//!    either of the reference implementation's backends.
//! 5. **Times**, recorded separately: each tower's encode, each path's
//!    language prefill of the image prompt, the decode steps; and prompt 1
//!    on Metal end to end (encode plus prefill) against the fork's 3.32 s
//!    prompt eval on the M3.
//! 6. **A 768 x 768 image** (the fixture resampled, 576 merged tokens):
//!    the Metal tower encodes it, the runner prefills the 595-row prompt and
//!    decodes a few greedy tokens, all finite — with the times printed.
//!
//! # Files
//!
//! Only `$OXI_BONSAI2_PQ2_GGUF`, `$OXI_BONSAI2_PTQ1_GGUF` and
//! `$OXI_BONSAI2_MMPROJ_GGUF` (or the release names under
//! `$OXIBONSAI_MODELS_DIR`) — never a workspace path, so an ordinary test
//! run maps no 27B; the goldens from `$OXI_BONSAI2_VISION_GOLDEN_DIR`, else
//! the vendored copy. Absent files skip with a `bonsai2-vision-metal` /
//! `executed: false` capability record (a failure under
//! `OXI_REQUIRE_MODEL_FILES=1`); `executed: true` is written only after
//! both bands passed. One band is mapped at a time.

use std::io::Write as _;
use std::path::PathBuf;
use std::time::Duration;

use oxibonsai_testkit::capability::{record_timed, Capability};

const PQ2_ENV: &str = "OXI_BONSAI2_PQ2_GGUF";
const PTQ1_ENV: &str = "OXI_BONSAI2_PTQ1_GGUF";
const MMPROJ_ENV: &str = "OXI_BONSAI2_MMPROJ_GGUF";
const MODELS_DIR_ENV: &str = "OXIBONSAI_MODELS_DIR";
const REQUIRE_ENV: &str = "OXI_REQUIRE_MODEL_FILES";
const PQ2_FILE: &str = "Ternary-Bonsai-2-27B-PQ2_0.gguf";
const PTQ1_FILE: &str = "Ternary-Bonsai-2-27B-PTQ1_0.gguf";
const MMPROJ_FILE: &str = "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf";
/// The capability these records are written under.
const CAPABILITY: Capability = Capability::Bonsai2VisionMetal;
const TEST: &str = "oxibonsai-model::bonsai2_vision_metal_tests::\
                    hybrid_real_27b_metal_vision_matches_the_cpu_path_and_the_fork_goldens_bonsai2";
fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var(name)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .map(PathBuf::from)
}

/// A release file from its variable or `$OXIBONSAI_MODELS_DIR` only.
fn locate(env: &str, file: &str) -> Option<PathBuf> {
    env_path(env)
        .or_else(|| env_path(MODELS_DIR_ENV).map(|dir| dir.join(file)))
        .filter(|p| p.is_file())
}

fn require_model_files() -> bool {
    std::env::var(REQUIRE_ENV).is_ok_and(|v| v.trim() == "1")
}

/// A line on the process's standard error that libtest does not capture,
/// so the evidence shows up in every run's log.
fn report(line: &str) {
    let mut stderr = std::io::stderr().lock();
    // Bookkeeping only: a failed diagnostic write must not fail the test.
    let _ = writeln!(stderr, "{line}");
}

/// Append one record to the capability manifest through the test kit (one
/// `write_all`, the documented schema); `duration_ms` is attached when the
/// gate measured one.
fn record(executed: bool, duration: Option<Duration>) {
    match duration {
        Some(d) => record_timed(CAPABILITY, executed, TEST, d),
        None => oxibonsai_testkit::capability::record(CAPABILITY, executed, TEST),
    }
    report(&format!(
        "CAPABILITY-REPORT capability={CAPABILITY} executed={executed} test={TEST}{}",
        duration.map_or_else(String::new, |d| format!(" duration_ms={}", d.as_millis()))
    ));
}

/// One golden step: the chosen id and the top-10.
#[derive(Debug, Clone)]
struct GoldenStep {
    id: u32,
    top: Vec<(u32, f64)>,
}

fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = logits.iter().map(|&l| f64::from(l - max).exp()).sum();
    let log_sum = sum.ln();
    logits
        .iter()
        .map(|&l| f64::from(l - max) - log_sum)
        .collect()
}

fn argmax(values: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &v) in values.iter().enumerate() {
        if v > values[best] {
            best = i;
        }
    }
    best as u32
}

/// The `n` most likely ids of a log-probability row, best first.
fn top_ids(logprobs: &[f64], n: usize) -> Vec<u32> {
    let mut ids: Vec<u32> = (0..logprobs.len() as u32).collect();
    let n = n.min(ids.len());
    ids.select_nth_unstable_by(n.saturating_sub(1), |&a, &b| {
        logprobs[b as usize].total_cmp(&logprobs[a as usize])
    });
    ids.truncate(n);
    ids.sort_by(|&a, &b| logprobs[b as usize].total_cmp(&logprobs[a as usize]));
    ids
}

/// The worse of two `|Δlogprob|` figures, NaN-sticky: once either is NaN the
/// result is NaN. `f64::max` returns the other operand when one is NaN, so a
/// fold with it drops a non-finite log-probability instead of carrying it to
/// the check made on the result; every worst-of accumulation of a distance in
/// this file folds with this.
fn worse_distance(so_far: f64, now: f64) -> f64 {
    if so_far.is_nan() || now.is_nan() {
        f64::NAN
    } else {
        so_far.max(now)
    }
}

/// Whether `distance` lies within `bound`. False for NaN: a NaN is within no
/// band. `distance > bound` is false for NaN as well, so a band check written
/// as that comparison lets a NaN pass; a check of a distance against its
/// band in this file asks `!within_band(…)` instead (the golden-ceiling
/// check pairs its `>` with an explicit `is_finite()`, to the same effect).
fn within_band(distance: f64, bound: f64) -> bool {
    distance <= bound
}

/// Worst `|Δlogprob|` of `ours` over a golden step's top-10; NaN when any
/// of those log-probabilities is NaN (never silently dropped).
fn worst_vs_golden(ours: &[f64], golden: &GoldenStep) -> f64 {
    golden
        .top
        .iter()
        .map(|&(id, lp)| (ours[id as usize] - lp).abs())
        .fold(0.0, worse_distance)
}

/// The golden's own top-1 / top-2 gap at a step.
fn golden_gap(golden: &GoldenStep) -> f64 {
    golden.top.first().map_or(0.0, |t| t.1) - golden.top.get(1).map_or(0.0, |t| t.1)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod real {
    use super::*;

    use std::path::Path;
    use std::time::Instant;

    const GOLDEN_ENV: &str = "OXI_BONSAI2_VISION_GOLDEN_DIR";
    /// Steps that must match the golden unconditionally.
    const MUST_MATCH: usize = 16;
    /// The G4 logprob band: the Metal and CPU paths' log-probabilities at every
    /// step.
    const BAND: f64 = 2.0e-2;
    /// The ceiling on each path's worst `|Δlogprob|` against each of the
    /// fork's goldens (`.metal` and `.cpu`) over the golden-context steps
    /// (module docs, point 4).
    const GOLDEN_CEILING: f64 = 1.25e-1;
    /// Log-probabilities compared per step: the top-N of each side.
    const TOP_N: usize = 10;
    /// The KV window both executors are built with: the 768 x 768 prompt's 595
    /// rows plus its decode steps, with room.
    const KV_WINDOW: usize = 1024;
    /// Greedy steps of the 768 x 768 smoke run.
    const SMOKE_STEPS: usize = 8;
    /// `<|im_end|>`, Bonsai 2's end-of-turn id.
    const IM_END: u32 = 248_046;

    fn golden_dir() -> PathBuf {
        env_path(GOLDEN_ENV).unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/bonsai2_golden_vision")
        })
    }

    #[derive(Debug, Clone)]
    struct Golden {
        steps: Vec<GoldenStep>,
        prompt_tokens: usize,
        finish: String,
    }

    fn load_golden(name: &str) -> Golden {
        let path = golden_dir().join(name);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("golden {}: {e}", path.display()));
        let value: serde_json::Value =
            serde_json::from_str(&text).unwrap_or_else(|e| panic!("golden {name}: {e}"));
        let response = &value["response"];
        let choice = &response["choices"][0];
        let steps = choice["logprobs"]["content"]
            .as_array()
            .unwrap_or_else(|| panic!("golden {name}: no logprobs"))
            .iter()
            .map(|step| GoldenStep {
                id: step["id"].as_u64().expect("golden id") as u32,
                top: step["top_logprobs"]
                    .as_array()
                    .expect("golden top_logprobs")
                    .iter()
                    .map(|t| {
                        (
                            t["id"].as_u64().expect("top id") as u32,
                            t["logprob"].as_f64().expect("top logprob"),
                        )
                    })
                    .collect(),
            })
            .collect();
        Golden {
            steps,
            prompt_tokens: response["usage"]["prompt_tokens"]
                .as_u64()
                .expect("prompt_tokens") as usize,
            finish: choice["finish_reason"]
                .as_str()
                .expect("finish_reason")
                .to_string(),
        }
    }

    fn load_average() -> String {
        std::process::Command::new("sysctl")
            .args(["-n", "vm.loadavg"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map_or_else(|| "unknown".to_string(), |s| s.trim().to_string())
    }

    use std::sync::Arc;

    use oxibonsai_core::gguf::reader::{mmap_gguf_file, GgufFile};
    use oxibonsai_kernels::KernelDispatcher;
    use oxibonsai_model::hybrid::metal::HybridMetalRunner;
    use oxibonsai_model::hybrid::{HybridModel, PromptPiece};
    use oxibonsai_model::vision::metal::VisionTowerMetal;
    use oxibonsai_model::vision::preprocess::resample;
    use oxibonsai_model::vision::{
        decode_image, plan_splice, prepare_image, GridSize, ImageRgb8, PreprocessConfig,
        ResizeFilter, SpliceSegment, VisionTokenIds, VisionTower, DEFAULT_IMAGE_MAX_TOKENS,
    };

    /// What the two towers made of the golden fixture.
    struct Encoded {
        cpu_rows: Vec<f32>,
        metal_rows: Vec<f32>,
        grid: GridSize,
        cpu_time: Duration,
        metal_time: Duration,
    }

    /// One path's run of one prompt: the greedy (or teacher-forced) ids,
    /// the raw logits of every step, and its times.
    struct PathRun {
        ids: Vec<u32>,
        logits: Vec<Vec<f32>>,
        rows: usize,
        prefill: Duration,
        decode: Duration,
    }

    fn metal_available() -> bool {
        match oxibonsai_kernels::MetalGraph::shared_device() {
            Ok(_) => true,
            Err(oxibonsai_kernels::MetalGraphError::DeviceNotFound) => false,
            Err(e) => panic!("the Metal device must open on this host: {e}"),
        }
    }

    pub(super) fn available() -> bool {
        metal_available()
    }

    fn fixture() -> ImageRgb8 {
        let png = std::fs::read(golden_dir().join("fixture_256x192.png")).expect("the fixture");
        decode_image(&png).expect("the fixture decodes")
    }

    /// Both towers on the golden fixture (already on the 32-pixel grid).
    fn encode_fixture(mmproj: &GgufFile<'_>, metal: &VisionTowerMetal) -> Encoded {
        let image = fixture();
        assert_eq!((image.width, image.height), (256, 192));
        let cfg = PreprocessConfig::qwen_vl(
            metal.config().patch_size,
            metal.config().spatial_merge,
            DEFAULT_IMAGE_MAX_TOKENS,
        )
        .expect("preprocess config");
        let prepared = prepare_image(&image, &cfg).expect("preprocess");
        assert_eq!(prepared.grid, GridSize { h: 6, w: 8 });
        assert_eq!(prepared.image, image, "the fixture is on the grid already");
        // The Metal tower: the first encode of a size also builds its
        // position rows; the second is the steady state.
        let t = Instant::now();
        let (first, _) = metal
            .encode(&prepared.image, DEFAULT_IMAGE_MAX_TOKENS)
            .expect("metal encode");
        let first_time = t.elapsed();
        let t = Instant::now();
        let (metal_rows, grid) = metal
            .encode(&prepared.image, DEFAULT_IMAGE_MAX_TOKENS)
            .expect("metal encode");
        let metal_time = t.elapsed();
        assert_eq!(metal_rows, first, "encodes of one image are bit-identical");
        let cpu = VisionTower::from_mmproj(mmproj).expect("the CPU tower loads");
        let t = Instant::now();
        let (cpu_rows, cpu_grid) = cpu
            .encode(&prepared.image, DEFAULT_IMAGE_MAX_TOKENS)
            .expect("cpu encode");
        let cpu_time = t.elapsed();
        assert_eq!(cpu_grid, grid);
        assert_eq!(grid, prepared.grid);
        report(&format!(
            "{CAPABILITY}: vision encode 256x192 -> {} x {} grid: metal {:.3}s (first {:.3}s), \
             cpu {:.2}s",
            grid.h,
            grid.w,
            metal_time.as_secs_f64(),
            first_time.as_secs_f64(),
            cpu_time.as_secs_f64()
        ));
        Encoded {
            cpu_rows,
            metal_rows,
            grid,
            cpu_time,
            metal_time,
        }
    }

    /// `tokens` with the image rows spliced at its one placeholder.
    fn pieces<'p>(tokens: &'p [u32], rows: &'p [f32], grid: GridSize) -> Vec<PromptPiece<'p>> {
        let plan = plan_splice(tokens, &[grid], VisionTokenIds::BONSAI2)
            .expect("one bracketed placeholder");
        plan.segments()
            .iter()
            .map(|segment| match segment {
                SpliceSegment::Text(range) => PromptPiece::Tokens(&tokens[range.clone()]),
                SpliceSegment::Image(_) => PromptPiece::Image { rows, grid },
            })
            .collect()
    }

    /// The CPU path: rows prefill, then `steps` greedy tokens.
    fn cpu_path(model: &mut HybridModel<'_>, pieces: &[PromptPiece<'_>], steps: usize) -> PathRun {
        let vocab = model.config().base.vocab_size;
        let mut logits = vec![0.0f32; vocab];
        model.reset();
        let t = Instant::now();
        let rows = model
            .forward_prefill_pieces(pieces, 0, Some(&mut logits))
            .expect("cpu prefill");
        let prefill = t.elapsed();
        let t = Instant::now();
        let mut ids = Vec::with_capacity(steps);
        let mut all = Vec::with_capacity(steps);
        for step in 0..steps {
            let id = argmax(&logits);
            ids.push(id);
            all.push(logits.clone());
            if step + 1 < steps {
                model
                    .forward(id, rows + step, &mut logits)
                    .expect("cpu decode");
            }
        }
        PathRun {
            ids,
            logits: all,
            rows,
            prefill,
            decode: t.elapsed(),
        }
    }

    /// The Metal path: the prompt assembled at the runner's rope start and
    /// prefilled by the runner, then decoded teacher-forced on `teacher`.
    fn metal_path(
        runner: &mut HybridMetalRunner<'_>,
        model: &HybridModel<'_>,
        pieces: &[PromptPiece<'_>],
        teacher: &[u32],
    ) -> PathRun {
        let vocab = runner.vocab_size();
        let mut logits = vec![0.0f32; vocab];
        runner.reset();
        let t = Instant::now();
        let assembled = model
            .assemble_prompt_at(pieces, runner.rope_start_for(0).expect("rope start"))
            .expect("assemble");
        runner
            .forward_prefill_rows(&assembled.rows, &assembled.positions, 0, Some(&mut logits))
            .expect("metal prefill");
        let prefill = t.elapsed();
        let rows = assembled.len();
        let t = Instant::now();
        let mut ids = Vec::with_capacity(teacher.len());
        let mut all = Vec::with_capacity(teacher.len());
        for (step, &next) in teacher.iter().enumerate() {
            ids.push(argmax(&logits));
            all.push(logits.clone());
            if step + 1 < teacher.len() {
                runner
                    .forward_into(next, rows + step, &mut logits)
                    .expect("metal decode");
            }
        }
        PathRun {
            ids,
            logits: all,
            rows,
            prefill,
            decode: t.elapsed(),
        }
    }

    /// Each path's worst `|Δlogprob|` against each of the fork's goldens,
    /// over the steps whose context is the golden's (module docs, point 4).
    #[derive(Debug, Clone, Copy, Default)]
    struct GoldenDistances {
        /// The Metal path against the `.metal` golden.
        metal_vs_metal: f64,
        /// The CPU path against the `.metal` golden.
        cpu_vs_metal: f64,
        /// The Metal path against the `.cpu` golden.
        metal_vs_cpu: f64,
        /// The CPU path against the `.cpu` golden.
        cpu_vs_cpu: f64,
    }

    impl GoldenDistances {
        /// Fold one step's four distances into the worst so far. A NaN
        /// distance (a non-finite log-probability) sticks ([`worse_distance`]),
        /// so the ceiling check sees it rather than `f64::max` dropping it.
        fn absorb(&mut self, step: Self) {
            self.metal_vs_metal = worse_distance(self.metal_vs_metal, step.metal_vs_metal);
            self.cpu_vs_metal = worse_distance(self.cpu_vs_metal, step.cpu_vs_metal);
            self.metal_vs_cpu = worse_distance(self.metal_vs_cpu, step.metal_vs_cpu);
            self.cpu_vs_cpu = worse_distance(self.cpu_vs_cpu, step.cpu_vs_cpu);
        }

        /// The four distances, labelled.
        fn labelled(self) -> [(&'static str, f64); 4] {
            [
                ("metal vs .metal", self.metal_vs_metal),
                ("cpu vs .metal", self.cpu_vs_metal),
                ("metal vs .cpu", self.metal_vs_cpu),
                ("cpu vs .cpu", self.cpu_vs_cpu),
            ]
        }
    }

    /// The figures of one prompt on one band.
    struct PromptResult {
        failures: Vec<String>,
        worst_metal_cpu: f64,
        worst_vs_goldens: GoldenDistances,
        matched: usize,
    }

    #[allow(clippy::too_many_arguments)]
    fn run_prompt(
        band: &str,
        label: &str,
        model: &mut HybridModel<'_>,
        runner: &mut HybridMetalRunner<'_>,
        tokens: &[u32],
        encoded: &Encoded,
        cpu_golden: &Golden,
        metal_golden: &Golden,
    ) -> PromptResult {
        let tag = format!("{CAPABILITY} {band} {label}");
        let steps = cpu_golden.steps.len();
        let cpu_pieces = pieces(tokens, &encoded.cpu_rows, encoded.grid);
        let metal_pieces = pieces(tokens, &encoded.metal_rows, encoded.grid);
        let cpu = cpu_path(model, &cpu_pieces, steps);
        let metal = metal_path(runner, model, &metal_pieces, &cpu.ids);
        let mut failures = Vec::new();
        if cpu.rows != cpu_golden.prompt_tokens || metal.rows != cpu_golden.prompt_tokens {
            failures.push(format!(
                "{tag}: prompt rows cpu {} / metal {}, the fork's usage.prompt_tokens {}",
                cpu.rows, metal.rows, cpu_golden.prompt_tokens
            ));
        }
        let delta = grid_delta(encoded.grid);
        if runner.rope_delta() != delta || model.rope_delta() != delta {
            failures.push(format!(
                "{tag}: M-RoPE offset runner {} / cpu {}, expected {delta}",
                runner.rope_delta(),
                model.rope_delta()
            ));
        }
        report(&format!(
            "{tag}: prompt {} tokens -> {} rows (48 image rows); language prefill cpu {:.2}s, \
             metal {:.3}s ({:.1}x); {steps} decode steps cpu {:.2}s, metal {:.2}s",
            tokens.len(),
            metal.rows,
            cpu.prefill.as_secs_f64(),
            metal.prefill.as_secs_f64(),
            cpu.prefill.as_secs_f64() / metal.prefill.as_secs_f64().max(1e-9),
            cpu.decode.as_secs_f64(),
            metal.decode.as_secs_f64()
        ));

        let mut result = PromptResult {
            failures: Vec::new(),
            worst_metal_cpu: 0.0,
            worst_vs_goldens: GoldenDistances::default(),
            matched: 0,
        };
        let mut on_golden = true;
        for step in 0..steps {
            let golden = &cpu_golden.steps[step];
            let lp_cpu = log_softmax(&cpu.logits[step]);
            let lp_metal = log_softmax(&metal.logits[step]);
            let mut ids = top_ids(&lp_cpu, TOP_N);
            ids.extend(golden.top.iter().map(|t| t.0));
            // NaN-sticky: a non-finite log-probability among these ids must
            // reach the band check below, not vanish into `f64::max`.
            let d_mc = ids
                .iter()
                .map(|&id| (lp_metal[id as usize] - lp_cpu[id as usize]).abs())
                .fold(0.0, worse_distance);
            result.worst_metal_cpu = worse_distance(result.worst_metal_cpu, d_mc);
            let gap = golden_gap(golden);
            let cpu_top2 = {
                let top = top_ids(&lp_cpu, 2);
                lp_cpu[top[0] as usize]
                    - top
                        .get(1)
                        .map_or(f64::NEG_INFINITY, |&i| lp_cpu[i as usize])
            };
            if metal.ids[step] != cpu.ids[step] {
                failures.push(format!(
                    "{tag} step {step}: metal {} != cpu {} (cpu top-2 gap {cpu_top2:.4}, \
                     |dlogprob| metal-cpu {d_mc:.3e})",
                    metal.ids[step], cpu.ids[step]
                ));
            }
            if !within_band(d_mc, BAND) {
                failures.push(format!(
                    "{tag} step {step}: |dlogprob| metal vs cpu {d_mc:.3e} is not within {BAND:e}"
                ));
            }
            // Against both goldens while the context is still the golden's
            // (the step that first diverges is still on the golden's
            // context: only its chosen token differs).
            let d = if on_golden {
                let metal_golden_step = metal_golden.steps.get(step).unwrap_or_else(|| {
                    panic!(
                        "{tag}: the .metal golden has no step {step} (the .cpu golden has \
                         {steps})"
                    )
                });
                let d = GoldenDistances {
                    metal_vs_metal: worst_vs_golden(&lp_metal, metal_golden_step),
                    cpu_vs_metal: worst_vs_golden(&lp_cpu, metal_golden_step),
                    metal_vs_cpu: worst_vs_golden(&lp_metal, golden),
                    cpu_vs_cpu: worst_vs_golden(&lp_cpu, golden),
                };
                result.worst_vs_goldens.absorb(d);
                d
            } else {
                GoldenDistances {
                    metal_vs_metal: f64::NAN,
                    cpu_vs_metal: f64::NAN,
                    metal_vs_cpu: f64::NAN,
                    cpu_vs_cpu: f64::NAN,
                }
            };
            let matches_golden = cpu.ids[step] == golden.id && metal.ids[step] == golden.id;
            if on_golden && matches_golden {
                result.matched += 1;
            }
            if on_golden && !matches_golden {
                let msg = format!(
                    "{tag} step {step}: golden {} vs cpu {} / metal {} (golden top-2 gap \
                     {gap:.4}; |dlogprob| vs .metal: metal {:.3e}, cpu {:.3e}; vs .cpu: metal \
                     {:.3e}, cpu {:.3e})",
                    golden.id,
                    cpu.ids[step],
                    metal.ids[step],
                    d.metal_vs_metal,
                    d.cpu_vs_metal,
                    d.metal_vs_cpu,
                    d.cpu_vs_cpu
                );
                report(&format!("DIVERGENCE {msg}"));
                if step < MUST_MATCH {
                    failures.push(msg);
                }
                on_golden = false;
            }
            report(&format!(
                "{tag} step {step:>2}: golden {:>6} cpu {:>6} metal {:>6} {} |dlp| metal-cpu \
                 {d_mc:.3e}; vs .metal: metal {:.3e} cpu {:.3e}; vs .cpu: metal {:.3e} cpu \
                 {:.3e}; golden top-2 gap {gap:.4}",
                golden.id,
                cpu.ids[step],
                metal.ids[step],
                if matches_golden { "ok  " } else { "DIFF" },
                d.metal_vs_metal,
                d.cpu_vs_metal,
                d.metal_vs_cpu,
                d.cpu_vs_cpu,
            ));
        }
        for (distance, worst) in result.worst_vs_goldens.labelled() {
            if worst > GOLDEN_CEILING || !worst.is_finite() {
                failures.push(format!(
                    "{tag}: worst |dlogprob| {distance} golden {worst:.3e} exceeds the ceiling \
                     {GOLDEN_CEILING:e} over the golden-context steps"
                ));
            }
        }
        if cpu_golden.finish == "stop" {
            assert_eq!(
                cpu_golden.steps.last().map(|s| s.id),
                Some(IM_END),
                "{tag}: a stopped golden ends with <|im_end|>"
            );
        }
        let worst = result.worst_vs_goldens;
        report(&format!(
            "{tag}: matched {}/{steps} golden ids on both paths; metal == cpu at every step: {}; \
             worst |dlogprob| metal vs cpu {:.3e} (band {BAND:e}); worst |dlogprob| vs the \
             goldens over the golden-context steps (ceiling {GOLDEN_CEILING:e}): metal vs .metal \
             {:.3e}, cpu vs .metal {:.3e}, metal vs .cpu {:.3e}, cpu vs .cpu {:.3e}",
            result.matched,
            metal.ids == cpu.ids,
            result.worst_metal_cpu,
            worst.metal_vs_metal,
            worst.cpu_vs_metal,
            worst.metal_vs_cpu,
            worst.cpu_vs_cpu
        ));
        result.failures = failures;
        result
    }

    /// `h * w - max(h, w)`: the M-RoPE offset one image leaves.
    fn grid_delta(grid: GridSize) -> usize {
        grid.n_tokens() - grid.h.max(grid.w)
    }

    /// The prompts through the GGUF's own chat template: an image part, then
    /// the question.
    fn render(gguf: &GgufFile<'_>, text: &str) -> String {
        use oxibonsai_tokenizer::chat_templates::{
            RenderContentPart, RenderMessage, RenderOptions, ResolvedChatTemplate,
        };
        let template = ResolvedChatTemplate::from_gguf(&gguf.metadata)
            .expect("the GGUF's own chat template compiles");
        template
            .render_with(
                &[RenderMessage::with_parts(
                    "user",
                    vec![
                        RenderContentPart::Image,
                        RenderContentPart::Text(text.to_string()),
                    ],
                )],
                &RenderOptions {
                    add_generation_prompt: true,
                    enable_thinking: Some(false),
                    ..RenderOptions::default()
                },
            )
            .expect("render")
    }

    /// One band: the CPU model and the runner over the same mapping, both
    /// prompts, then the 768 x 768 smoke run.
    fn run_band(
        band: &str,
        model_path: &Path,
        encoded: &Encoded,
        metal_tower: &VisionTowerMetal,
        goldens: &[(Golden, Golden); 2],
    ) -> Vec<String> {
        let map = mmap_gguf_file(model_path).expect("map the 27B");
        let gguf = GgufFile::parse(&map).expect("parse the 27B");
        let config = HybridModel::config_from_gguf(&gguf).expect("qwen35 config");
        let kernel = Arc::new(KernelDispatcher::with_tier(
            oxibonsai_kernels::cpu_kernel_tier(),
        ));
        let loading = Instant::now();
        let mut model =
            HybridModel::from_gguf_with(&gguf, config, KV_WINDOW, &kernel).expect("27B loads");
        let mut runner =
            HybridMetalRunner::new_in_place(&model, gguf.data).expect("the Metal runner builds");
        report(&format!(
            "{CAPABILITY} {band}: model and runner loaded in {:.2}s (weights mapped in place: \
             {}); load average {}",
            loading.elapsed().as_secs_f64(),
            runner.is_mapped(),
            load_average()
        ));
        assert_eq!(
            encoded.metal_rows.len(),
            encoded.grid.n_tokens() * model.config().base.hidden_size
        );
        let tokenizer = oxibonsai_tokenizer::OxiTokenizer::from_gguf_metadata(&gguf.metadata)
            .expect("tokenizer");
        let prompts = [
            ("prompt1", render(&gguf, "Describe this image briefly.")),
            (
                "prompt2",
                render(
                    &gguf,
                    "What shapes and colors do you see? Answer in one sentence.",
                ),
            ),
        ];
        let mut failures = Vec::new();
        for ((label, rendered), (cpu_golden, metal_golden)) in prompts.iter().zip(goldens) {
            let tokens = tokenizer.encode(rendered).expect("tokenize");
            let result = run_prompt(
                band,
                label,
                &mut model,
                &mut runner,
                &tokens,
                encoded,
                cpu_golden,
                metal_golden,
            );
            failures.extend(result.failures);
            if *label == "prompt1" {
                // Prompt 1 on Metal end to end: the Metal tower's encode and
                // the runner's prefill of the 67 rows.
                runner.reset();
                let metal_pieces = pieces(&tokens, &encoded.metal_rows, encoded.grid);
                let t = Instant::now();
                let assembled = model
                    .assemble_prompt_at(&metal_pieces, 0)
                    .expect("assemble");
                let mut logits = vec![0.0f32; runner.vocab_size()];
                runner
                    .forward_prefill_rows(
                        &assembled.rows,
                        &assembled.positions,
                        0,
                        Some(&mut logits),
                    )
                    .expect("prefill");
                let prefill = t.elapsed();
                report(&format!(
                    "{CAPABILITY} {band} prompt1 end to end on Metal: encode {:.3}s + prefill of \
                     {} rows {:.3}s = {:.3}s (the fork's prompt eval: 3.32s; informational target \
                     5s)",
                    encoded.metal_time.as_secs_f64(),
                    assembled.len(),
                    prefill.as_secs_f64(),
                    (encoded.metal_time + prefill).as_secs_f64()
                ));
            }
        }
        failures.extend(smoke_768(
            band,
            &mut runner,
            &model,
            metal_tower,
            &tokenizer,
            &gguf,
        ));
        failures
    }

    /// The fixture resampled to 768 x 768 (576 merged tokens) through the
    /// Metal tower and the runner: prefill and a few greedy steps, finite.
    fn smoke_768(
        band: &str,
        runner: &mut HybridMetalRunner<'_>,
        model: &HybridModel<'_>,
        tower: &VisionTowerMetal,
        tokenizer: &oxibonsai_tokenizer::OxiTokenizer,
        gguf: &GgufFile<'_>,
    ) -> Vec<String> {
        let tag = format!("{CAPABILITY} {band} 768x768");
        let image = resample(&fixture(), 768, 768, ResizeFilter::Bicubic).expect("resample");
        let t = Instant::now();
        let (rows, grid) = tower
            .encode(&image, DEFAULT_IMAGE_MAX_TOKENS)
            .expect("metal encode 768x768");
        let encode = t.elapsed();
        let tokens = tokenizer
            .encode(&render(gguf, "Describe this image briefly."))
            .expect("tokenize");
        let prompt = pieces(&tokens, &rows, grid);
        runner.reset();
        let t = Instant::now();
        let assembled = model.assemble_prompt_at(&prompt, 0).expect("assemble");
        let mut logits = vec![0.0f32; runner.vocab_size()];
        runner
            .forward_prefill_rows(&assembled.rows, &assembled.positions, 0, Some(&mut logits))
            .expect("prefill 768x768");
        let prefill = t.elapsed();
        let t = Instant::now();
        let mut ids = Vec::new();
        let mut finite = logits.iter().all(|v| v.is_finite());
        for step in 0..SMOKE_STEPS {
            let id = argmax(&logits);
            ids.push(id);
            runner
                .forward_into(id, assembled.len() + step, &mut logits)
                .expect("decode");
            finite &= logits.iter().all(|v| v.is_finite());
        }
        let decode = t.elapsed();
        let text = tokenizer.decode(&ids).unwrap_or_default();
        report(&format!(
            "{tag}: {} x {} grid ({} image rows, {} prompt rows); encode {:.3}s, prefill {:.3}s \
             ({:.1} ms/row), {SMOKE_STEPS} greedy steps {:.2}s: {text:?}",
            grid.h,
            grid.w,
            grid.n_tokens(),
            assembled.len(),
            encode.as_secs_f64(),
            prefill.as_secs_f64(),
            prefill.as_secs_f64() * 1e3 / assembled.len() as f64,
            decode.as_secs_f64()
        ));
        let mut failures = Vec::new();
        if grid != (GridSize { h: 24, w: 24 }) {
            failures.push(format!("{tag}: grid {grid:?}, expected 24 x 24"));
        }
        if !finite {
            failures.push(format!("{tag}: non-finite logits"));
        }
        failures
    }

    pub(super) fn run(bands: &[(&str, PathBuf)], mmproj_path: &Path) {
        let started = Instant::now();
        let goldens = [
            (
                load_golden("vision.prompt1.cpu.json"),
                load_golden("vision.prompt1.metal.json"),
            ),
            (
                load_golden("vision.prompt2.cpu.json"),
                load_golden("vision.prompt2.metal.json"),
            ),
        ];
        for (cpu, metal) in &goldens {
            assert_eq!(
                cpu.steps.iter().map(|s| s.id).collect::<Vec<_>>(),
                metal.steps.iter().map(|s| s.id).collect::<Vec<_>>(),
                "the fork's Metal and CPU goldens agree on the ids"
            );
        }
        assert_eq!(
            (goldens[0].0.prompt_tokens, goldens[1].0.prompt_tokens),
            (67, 75),
            "the fork's usage.prompt_tokens"
        );

        let mmproj_map = mmap_gguf_file(mmproj_path).expect("map the projector");
        let mmproj = GgufFile::parse(&mmproj_map).expect("parse the projector");
        let loading = Instant::now();
        let tower = VisionTowerMetal::from_mmproj(&mmproj, DEFAULT_IMAGE_MAX_TOKENS)
            .expect("the Metal tower loads");
        report(&format!(
            "{CAPABILITY}: Metal tower loaded in {:.2}s, {} bytes resident",
            loading.elapsed().as_secs_f64(),
            tower.resident_bytes()
        ));
        let encoded = encode_fixture(&mmproj, &tower);
        report(&format!(
            "{CAPABILITY}: vision encode (separately from the language prefill): metal {:.3}s, \
             cpu {:.2}s",
            encoded.metal_time.as_secs_f64(),
            encoded.cpu_time.as_secs_f64()
        ));

        let mut failures = Vec::new();
        for (band, path) in bands {
            failures.extend(run_band(band, path, &encoded, &tower, &goldens));
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
        record(true, Some(started.elapsed()));
    }
}

/// Both bands of the 27B with the projector on Metal, against the CPU path
/// and the fork's goldens (see the module docs).
#[test]
fn hybrid_real_27b_metal_vision_matches_the_cpu_path_and_the_fork_goldens_bonsai2() {
    let located = (
        locate(PQ2_ENV, PQ2_FILE),
        locate(PTQ1_ENV, PTQ1_FILE),
        locate(MMPROJ_ENV, MMPROJ_FILE),
    );
    let (Some(pq2), Some(ptq1), Some(mmproj)) = located else {
        assert!(
            !require_model_files(),
            "{REQUIRE_ENV}=1 but {PQ2_FILE}, {PTQ1_FILE} and {MMPROJ_FILE} were not all found \
             (set {PQ2_ENV}, {PTQ1_ENV} and {MMPROJ_ENV}, or {MODELS_DIR_ENV})"
        );
        report(&format!(
            "skip {TEST}: set {PQ2_ENV}, {PTQ1_ENV} and {MMPROJ_ENV} (or {MODELS_DIR_ENV})"
        ));
        record(false, None);
        return;
    };
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        if !real::available() {
            assert!(
                !require_model_files(),
                "{REQUIRE_ENV}=1 but no Metal device"
            );
            record(false, None);
            return;
        }
        real::run(&[("PQ2_0", pq2), ("PTQ1_0", ptq1)], &mmproj);
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        let _ = (pq2, ptq1, mmproj);
        assert!(
            !require_model_files(),
            "{REQUIRE_ENV}=1 but this build has no Metal backend"
        );
        record(false, None);
    }
}

/// The helpers' arithmetic on hand-made rows.
#[test]
fn log_softmax_and_the_top_ids_rank_a_row() {
    let logits = [1.0f32, 3.0, 2.0, -1.0];
    let lp = log_softmax(&logits);
    let total: f64 = lp.iter().map(|v| v.exp()).sum();
    assert!((total - 1.0).abs() < 1e-12);
    assert_eq!(top_ids(&lp, 2), vec![1, 2]);
    assert_eq!(top_ids(&lp, 9), vec![1, 2, 0, 3]);
    assert_eq!(argmax(&logits), 1);
    let golden = GoldenStep {
        id: 1,
        top: vec![(1, lp[1] + 0.01), (2, lp[2] - 0.02)],
    };
    assert_eq!(top_ids(&lp, 1), vec![golden.id]);
    assert!((worst_vs_golden(&lp, &golden) - 0.02).abs() < 1e-12);
    assert!((golden_gap(&golden) - (lp[1] - lp[2] + 0.03)).abs() < 1e-12);
    // A NaN log-probability among the golden's top ids is not dropped.
    let mut poisoned = lp.clone();
    poisoned[2] = f64::NAN;
    assert!(worst_vs_golden(&poisoned, &golden).is_nan());
}

/// The band arithmetic on hand-made distances: a worst-of fold carries a NaN
/// wherever it sits, and a NaN is within no band.
#[test]
fn a_nan_distance_sticks_in_the_fold_and_fails_every_band() {
    // Ordinary distances: the fold is their maximum.
    assert_eq!([0.5, 2.0, 1.0].into_iter().fold(0.0, worse_distance), 2.0);
    assert_eq!(worse_distance(0.0, 0.0), 0.0);
    // A NaN sticks in any position, as the accumulator or as the next term;
    // the plain `f64::max` fold of the same terms drops it, which is what
    // this helper exists to prevent.
    for terms in [
        [f64::NAN, 1.0, 2.0],
        [1.0, f64::NAN, 2.0],
        [1.0, 2.0, f64::NAN],
    ] {
        assert!(
            terms.into_iter().fold(0.0, worse_distance).is_nan(),
            "{terms:?}"
        );
        assert!(
            !terms.into_iter().fold(0.0, f64::max).is_nan(),
            "{terms:?}: the plain fold would have dropped the NaN"
        );
    }
    assert!(worse_distance(f64::NAN, 1.0).is_nan());
    assert!(worse_distance(1.0, f64::NAN).is_nan());
    // The band: on the bound is within, past it is not, and a NaN never is.
    let band = 2.0e-2;
    assert!(within_band(0.0, band));
    assert!(within_band(band, band));
    assert!(!within_band(band * 1.000_001, band));
    assert!(!within_band(f64::INFINITY, band));
    assert!(!within_band(f64::NAN, band));
    // A NaN is not above the band either — a check written as
    // `distance > band` passes it. (Through `black_box`, so the comparison is
    // made at run time rather than rejected as a comparison with a constant
    // NaN.)
    let nan = std::hint::black_box(f64::NAN);
    let nan_is_above = nan > band;
    assert!(!nan_is_above);
    // The two together, as `run_prompt` uses them: a NaN step among clean
    // ones fails the band.
    let d_mc = [1.0e-3, f64::NAN, 5.0e-3]
        .into_iter()
        .fold(0.0, worse_distance);
    assert!(!within_band(d_mc, band));
}
