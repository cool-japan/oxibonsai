//! Real-model vision gate (bonsai2-design.md §6.2 / §8.2): the Bonsai 2 27B
//! `PQ2_0` language model and its Qwen3-VL projector, end to end on the CPU
//! path, against the PrismML fork's own goldens.
//!
//! # The oracle
//!
//! `tests/fixtures/bonsai2_golden_vision/` holds the fork's
//! `llama-server --mmproj` answers (`/v1/chat/completions`, greedy, 32
//! tokens, top-10 logprobs, `enable_thinking = false`) for two prompts about
//! one synthetic 256 x 192 image (`fixture_256x192.png`), once with every
//! layer on Metal (`.metal`) and once on the CPU (`.cpu`) — both backends
//! produced the same ids, so the golden is backend-independent — plus the
//! fork's `apply_template` rendering of prompt 1. The one path element of
//! the captures (the capture machine's model path in the `model` field) is
//! reduced to the release file name.
//!
//! # What is checked
//!
//! 1. The rendered prompt (the fork's own `apply_template` text, its media
//!    marker standing where the template's
//!    `<|vision_start|><|image_pad|><|vision_end|>` goes) tokenizes to one
//!    bracketed `<|image_pad|>`, which expands to exactly
//!    `(192 / 32) * (256 / 32) = 48` image rows, for 67 / 75 prompt rows —
//!    the fork server's own `usage.prompt_tokens`.
//! 2. The fixture decodes, preprocesses to its own size (no resampling:
//!    256 x 192 is already on the 32-pixel grid) and encodes to a 6 x 8
//!    merged grid.
//! 3. Greedy decoding, teacher-forced on the golden ids: our top-1 equals
//!    the golden token at every step (and the first 16 unconditionally);
//!    prompt 2's golden ends with the `<|im_end|>` the fork stopped on, so
//!    matching it is matching the stop. Where a step ever disagrees, the
//!    report names it and the golden's own top-2 gap there.
//! 4. Informational: the worst `|Δlogprob|` over the golden top-10 against
//!    the `.cpu` and the `.metal` goldens (for scale: the fork's own
//!    CPU-vs-Metal spread on these goldens is 6.8e-2 / 6.4e-2).
//!
//! # Files
//!
//! `$OXI_BONSAI2_PQ2_GGUF` / `$OXI_BONSAI2_MMPROJ_GGUF`, else the release
//! file names under `$OXIBONSAI_MODELS_DIR` or the workspace `models/`
//! directory (testkit resolver). Absent files skip with a `bonsai2-vision`
//! / `executed: false` capability record — unless
//! `OXI_REQUIRE_MODEL_FILES=1`, which turns the absence into a failure.
//! `executed: true` is written only after every assertion passed.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use oxibonsai_core::gguf::reader::{mmap_gguf_file, GgufFile};
use oxibonsai_kernels::KernelDispatcher;
use oxibonsai_model::hybrid::{HybridModel, PromptPiece};
use oxibonsai_model::vision::{
    decode_image, plan_splice, prepare_image, GridSize, PreprocessConfig, SpliceSegment,
    VisionTokenIds, VisionTower, DEFAULT_IMAGE_MAX_TOKENS,
};
use oxibonsai_testkit::capability::{record_timed, Capability};

const PQ2_ENV: &str = "OXI_BONSAI2_PQ2_GGUF";
const MMPROJ_ENV: &str = "OXI_BONSAI2_MMPROJ_GGUF";
const MODELS_DIR_ENV: &str = "OXIBONSAI_MODELS_DIR";
const REQUIRE_ENV: &str = "OXI_REQUIRE_MODEL_FILES";
const PQ2_FILE: &str = "Ternary-Bonsai-2-27B-PQ2_0.gguf";
const MMPROJ_FILE: &str = "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf";
/// The capability these records are written under.
const CAPABILITY: Capability = Capability::Bonsai2Vision;
const TEST: &str =
    "oxibonsai-model::bonsai2_vision_tests::bonsai2_vision_real_27b_matches_the_fork_goldens_on_the_cpu_path";
/// The fork's media marker in its `apply_template` output.
const MEDIA_MARKER_PREFIX: &str = "<__media_";
/// What the template renders an image part as (the fork's `mtmd_tokenize`
/// expands its marker into the same bracket).
const IMAGE_PLACEHOLDER: &str = "<|vision_start|><|image_pad|><|vision_end|>";
/// `<|im_end|>`, Bonsai 2's end-of-turn id.
const IM_END: u32 = 248_046;
/// Steps that must match unconditionally.
const MUST_MATCH: usize = 16;
const KV_WINDOW: usize = 512;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/bonsai2_golden_vision")
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var(name)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .map(PathBuf::from)
}

fn locate(env: &str, file: &str) -> Option<PathBuf> {
    env_path(env)
        .or_else(|| env_path(MODELS_DIR_ENV).map(|dir| dir.join(file)))
        .filter(|p| p.is_file())
        .or_else(|| oxibonsai_testkit::workspace::find_model(file))
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

/// Append one record to the capability manifest through the testkit (one
/// `write_all`, the documented schema); `duration_ms` is attached when the
/// gate measured one.
fn record(executed: bool, duration: Option<std::time::Duration>) {
    match duration {
        Some(d) => record_timed(CAPABILITY, executed, TEST, d),
        None => oxibonsai_testkit::capability::record(CAPABILITY, executed, TEST),
    }
    report(&format!(
        "CAPABILITY-REPORT capability={CAPABILITY} executed={executed} test={TEST}{}",
        duration.map_or_else(String::new, |d| format!(" duration_ms={}", d.as_millis()))
    ));
}

/// One golden step: the chosen id and logprob, and the top-10.
struct GoldenStep {
    id: u32,
    top: Vec<(u32, f64)>,
}

struct Golden {
    steps: Vec<GoldenStep>,
    prompt_tokens: usize,
    finish: String,
}

fn load_golden(name: &str) -> Golden {
    let path = fixture_dir().join(name);
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("golden {}: {e}", path.display()));
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

/// The fork's `apply_template` rendering of prompt 1, its media marker
/// replaced by the template's image placeholder.
fn rendered_prompt1() -> String {
    let path = fixture_dir().join("apply_template.vision.metal.json");
    let text = std::fs::read_to_string(&path).expect("apply_template golden");
    let value: serde_json::Value = serde_json::from_str(&text).expect("apply_template json");
    let prompt = value["prompt"].as_str().expect("prompt").to_string();
    let start = prompt.find(MEDIA_MARKER_PREFIX).expect("media marker");
    let end = start + prompt[start..].find("__>").expect("marker end") + 3;
    format!("{}{IMAGE_PLACEHOLDER}{}", &prompt[..start], &prompt[end..])
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

/// Worst `|Δlogprob|` over a golden step's top-10.
fn worst_delta(ours: &[f64], golden: &GoldenStep) -> f64 {
    golden
        .top
        .iter()
        .map(|&(id, lp)| (ours[id as usize] - lp).abs())
        .fold(0.0, f64::max)
}

struct PromptResult {
    matched_prefix: usize,
    first_divergence: Option<(usize, u32, u32, f64)>,
    worst_cpu: f64,
    worst_metal: f64,
    /// Expanded prompt rows (text tokens plus image rows).
    rows: usize,
    /// The language model's prefill of those rows, alone.
    prefill: std::time::Duration,
    /// The teacher-forced decode steps.
    decode: std::time::Duration,
}

#[allow(clippy::too_many_arguments)]
fn run_prompt(
    label: &str,
    model: &mut HybridModel<'_>,
    tokenizer: &oxibonsai_tokenizer::OxiTokenizer,
    rendered: &str,
    image_rows: &[f32],
    grid: GridSize,
    cpu: &Golden,
    metal: &Golden,
) -> PromptResult {
    let tokens = tokenizer.encode(rendered).expect("tokenize");
    let ids = VisionTokenIds::BONSAI2;
    let plan = plan_splice(&tokens, &[grid], ids).expect("one bracketed placeholder");
    assert_eq!(
        plan.total_rows(),
        cpu.prompt_tokens,
        "{label}: expanded prompt rows must equal the fork's usage.prompt_tokens"
    );
    assert_eq!(plan.image_rows(), grid.n_tokens());
    let mut pieces = Vec::new();
    for segment in plan.segments() {
        match segment {
            SpliceSegment::Text(range) => pieces.push(PromptPiece::Tokens(&tokens[range.clone()])),
            SpliceSegment::Image(_) => pieces.push(PromptPiece::Image {
                rows: image_rows,
                grid,
            }),
        }
    }
    let vocab = model.config().base.vocab_size;
    let mut logits = vec![0.0f32; vocab];
    model.reset();
    let t0 = Instant::now();
    let rows = model
        .forward_prefill_pieces(&pieces, 0, Some(&mut logits))
        .unwrap_or_else(|e| panic!("{label}: prefill: {e}"));
    let prefill = t0.elapsed();
    assert_eq!(rows, cpu.prompt_tokens);
    assert_eq!(
        model.rope_delta(),
        grid.n_tokens() - grid.h.max(grid.w),
        "{label}: M-RoPE offset after one image"
    );
    report(&format!(
        "{CAPABILITY} {label}: prompt {} tokens -> {rows} rows (48 image rows), language prefill {:.2}s",
        tokens.len(),
        prefill.as_secs_f64()
    ));

    let mut result = PromptResult {
        matched_prefix: 0,
        first_divergence: None,
        worst_cpu: 0.0,
        worst_metal: 0.0,
        rows,
        prefill,
        decode: std::time::Duration::ZERO,
    };
    let t1 = Instant::now();
    let mut all_matched = true;
    for (step, golden) in cpu.steps.iter().enumerate() {
        let ours = argmax(&logits);
        let logprobs = log_softmax(&logits);
        let d_cpu = worst_delta(&logprobs, golden);
        let d_metal = metal
            .steps
            .get(step)
            .map_or(0.0, |g| worst_delta(&logprobs, g));
        let gap = golden.top.first().map_or(0.0, |t| t.1) - golden.top.get(1).map_or(0.0, |t| t.1);
        if ours == golden.id {
            if all_matched {
                result.matched_prefix += 1;
                result.worst_cpu = result.worst_cpu.max(d_cpu);
                result.worst_metal = result.worst_metal.max(d_metal);
            }
        } else {
            all_matched = false;
            if result.first_divergence.is_none() {
                result.first_divergence = Some((step, golden.id, ours, gap));
            }
        }
        report(&format!(
            "{CAPABILITY} {label} step {step:>2}: golden {:>6} ours {:>6} {} |dlp| cpu {d_cpu:.3e} metal {d_metal:.3e} golden top-2 gap {gap:.4}",
            golden.id,
            ours,
            if ours == golden.id { "ok " } else { "DIFF" },
        ));
        // Teacher-forced: the next step sees the golden token either way.
        let pos = rows + step;
        model
            .forward(golden.id, pos, &mut logits)
            .unwrap_or_else(|e| panic!("{label}: decode at {pos}: {e}"));
    }
    let decode = t1.elapsed();
    result.decode = decode;
    report(&format!(
        "{CAPABILITY} {label}: {} decode steps in {:.2}s",
        cpu.steps.len(),
        decode.as_secs_f64()
    ));
    if cpu.finish == "stop" {
        // The fork reports the terminating `<|im_end|>` as the last step of
        // a stopped completion; matching it above is matching the stop.
        assert_eq!(
            cpu.steps.last().map(|s| s.id),
            Some(IM_END),
            "{label}: a stopped golden ends with <|im_end|>"
        );
        report(&format!(
            "{CAPABILITY} {label}: <|im_end|> at step {} ({}), where the fork stopped",
            cpu.steps.len() - 1,
            if result.first_divergence.is_none() {
                "matched"
            } else {
                "NOT matched"
            }
        ));
    }
    result
}

#[test]
fn bonsai2_vision_real_27b_matches_the_fork_goldens_on_the_cpu_path() {
    let located = (locate(PQ2_ENV, PQ2_FILE), locate(MMPROJ_ENV, MMPROJ_FILE));
    let (Some(model_path), Some(mmproj_path)) = located else {
        assert!(
            !require_model_files(),
            "{REQUIRE_ENV}=1 but {PQ2_FILE} / {MMPROJ_FILE} were not found (set {PQ2_ENV} and \
             {MMPROJ_ENV})"
        );
        record(false, None);
        return;
    };
    let started = Instant::now();

    // The fixture: decoded, preprocessed exactly like the fork (already on
    // the grid, so untouched), encoded by the real projector.
    let png = std::fs::read(fixture_dir().join("fixture_256x192.png")).expect("fixture");
    let image = decode_image(&png).expect("fixture decodes");
    assert_eq!((image.width, image.height), (256, 192));
    let mmproj_map = mmap_gguf_file(&mmproj_path).expect("mmap mmproj");
    let mmproj = GgufFile::parse(&mmproj_map).expect("parse mmproj");
    let tower = VisionTower::from_mmproj(&mmproj).expect("vision tower loads");
    let cfg = PreprocessConfig::for_tower(&tower, DEFAULT_IMAGE_MAX_TOKENS).expect("config");
    let prepared = prepare_image(&image, &cfg).expect("preprocess");
    assert_eq!(prepared.grid, GridSize { h: 6, w: 8 });
    assert_eq!(
        prepared.image, image,
        "the fixture is on the 32-pixel grid already"
    );
    let t_encode = Instant::now();
    let (rows, grid) = tower
        .encode(&prepared.image, DEFAULT_IMAGE_MAX_TOKENS)
        .expect("encode");
    let encode = t_encode.elapsed();
    assert_eq!(grid, prepared.grid);
    assert_eq!(grid.n_tokens(), (192 / 32) * (256 / 32));
    report(&format!(
        "{CAPABILITY}: vision encode 256x192 -> {} x {} merged grid ({} rows x {}) in {:.2}s",
        grid.h,
        grid.w,
        grid.n_tokens(),
        rows.len() / grid.n_tokens(),
        encode.as_secs_f64()
    ));
    drop(tower);

    // The language model on the CPU tier.
    let map = mmap_gguf_file(&model_path).expect("mmap 27B");
    let gguf = GgufFile::parse(&map).expect("parse 27B");
    let config = HybridModel::config_from_gguf(&gguf).expect("qwen35 config");
    let kernel = Arc::new(KernelDispatcher::with_tier(
        oxibonsai_kernels::cpu_kernel_tier(),
    ));
    let mut model =
        HybridModel::from_gguf_with(&gguf, config, KV_WINDOW, &kernel).expect("27B PQ2_0 loads");
    assert_eq!(
        rows.len(),
        grid.n_tokens() * model.config().base.hidden_size
    );
    let tokenizer =
        oxibonsai_tokenizer::OxiTokenizer::from_gguf_metadata(&gguf.metadata).expect("tokenizer");

    // The vocabulary's own vision ids are the ones the splice keys on.
    let ids = VisionTokenIds::BONSAI2;
    for (text, id) in [
        ("<|vision_start|>", ids.vision_start),
        ("<|vision_end|>", ids.vision_end),
        ("<|image_pad|>", ids.image_pad),
        ("<|im_end|>", IM_END),
    ] {
        assert_eq!(tokenizer.encode(text).expect("encode"), vec![id], "{text}");
    }

    // The prompts through the GGUF's own chat template, with an image part
    // and a text part — rendered exactly as the fork's `apply_template` did
    // (its media marker is the template's placeholder), token for token.
    let template =
        oxibonsai_tokenizer::chat_templates::ResolvedChatTemplate::from_gguf(&gguf.metadata)
            .expect("the GGUF's own chat template compiles");
    let render = |text: &str| {
        use oxibonsai_tokenizer::chat_templates::{
            RenderContentPart, RenderMessage, RenderOptions,
        };
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
    };
    let prompt1 = render("Describe this image briefly.");
    let prompt2 = render("What shapes and colors do you see? Answer in one sentence.");
    let fork_prompt1 = rendered_prompt1();
    assert_eq!(
        prompt1, fork_prompt1,
        "our rendering vs the fork's apply_template"
    );
    assert_eq!(
        tokenizer.encode(&prompt1).expect("encode"),
        tokenizer.encode(&fork_prompt1).expect("encode")
    );
    assert!(prompt1.contains(IMAGE_PLACEHOLDER) && prompt2.contains(IMAGE_PLACEHOLDER));
    report(&format!(
        "{CAPABILITY}: the rendered prompt equals the fork's apply_template output ({} tokens \
         before expansion)",
        tokenizer.encode(&prompt1).map(|t| t.len()).unwrap_or(0)
    ));

    let mut failures = Vec::new();
    for (label, rendered, cpu_name, metal_name) in [
        (
            "prompt1",
            &prompt1,
            "vision.prompt1.cpu.json",
            "vision.prompt1.metal.json",
        ),
        (
            "prompt2",
            &prompt2,
            "vision.prompt2.cpu.json",
            "vision.prompt2.metal.json",
        ),
    ] {
        let cpu = load_golden(cpu_name);
        let metal = load_golden(metal_name);
        assert_eq!(
            cpu.steps.iter().map(|s| s.id).collect::<Vec<_>>(),
            metal.steps.iter().map(|s| s.id).collect::<Vec<_>>(),
            "the fork's Metal and CPU goldens agree"
        );
        let result = run_prompt(
            label, &mut model, &tokenizer, rendered, &rows, grid, &cpu, &metal,
        );
        report(&format!(
            "{CAPABILITY} golden {label}: matched {}/{} greedy ids; worst |dlogprob| over the \
             matched prefix: cpu {:.3e}, metal {:.3e}; first divergence {}",
            result.matched_prefix,
            cpu.steps.len(),
            result.worst_cpu,
            result.worst_metal,
            result.first_divergence.map_or_else(
                || "none".to_string(),
                |(step, golden, ours, gap)| format!(
                    "step {step} (golden {golden}, ours {ours}, golden top-2 gap {gap:.4})"
                )
            )
        ));
        // The vision encode (shared by both prompts) and each prompt's
        // language prefill, timed separately.
        report(&format!(
            "{CAPABILITY} golden {label} timing: vision encode {:.2}s (once, {} image rows); \
             language prefill {:.2}s for {} rows; {} decode steps {:.2}s",
            encode.as_secs_f64(),
            grid.n_tokens(),
            result.prefill.as_secs_f64(),
            result.rows,
            cpu.steps.len(),
            result.decode.as_secs_f64()
        ));
        if result.matched_prefix < MUST_MATCH.min(cpu.steps.len()) {
            failures.push(format!(
                "{label}: only {} of the first {MUST_MATCH} ids match",
                result.matched_prefix
            ));
        }
        if let Some((step, golden, ours, gap)) = result.first_divergence {
            failures.push(format!(
                "{label}: top-1 differs from the golden at step {step} (golden {golden}, ours \
                 {ours}, golden top-2 gap {gap:.4})"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("; "));
    record(true, Some(started.elapsed()));
}
