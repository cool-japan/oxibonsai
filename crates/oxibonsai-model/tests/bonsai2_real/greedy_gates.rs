//! Design §8.2 **G1** and **G4** on the real Bonsai 2 27B, against the
//! PrismML fork's own goldens (B2-11-FIX; ruling R2' of
//! `pkg/wave4b.rulings.md`).
//!
//! # What is asserted
//!
//! `PQ2_0` (`OXI_BONSAI2_PQ2_GGUF`), all three raw prompts × 24 steps,
//! against `golden2/PQ2_0.prompt{i}.server.json` (the fork on **Metal**):
//!
//! 1. **G1** — our greedy token equals the fork's at every step (hard);
//! 2. **G4** — our top-10 id *set* equals the fork's at every step (hard);
//!    the *order* is recorded, not asserted (the fork's own two backends
//!    disagree on top-10 order at 11 of 72 steps);
//! 3. **G4** — the worst `|Δlogprob|` over the common top-10 is at most
//!    [`G4_LOGPROB_BAND`] (2.0e-2, absolute) at every step (hard) — below
//!    the fork's own Metal-vs-CPU spread;
//! 4. the decoded 32-token continuation is byte-identical to the fork's
//!    `llama-cli` text dump (`Ternary-Bonsai-2-27B-PQ2_0.prompt{i}.txt`,
//!    which ends with the tool's trailing `"\n\n"`).
//!
//! The same table is computed against `golden_cpu/PQ2_0.prompt{i}.server.json`
//! (the fork on **CPU**) and printed — informational, per the ruling.
//!
//! `PTQ1_0` (`OXI_BONSAI2_PTQ1_GGUF`): **G1** against the `PQ2_0` greedy ids
//! (design §7.0.2: the fork's two bands produce byte-identical output on all
//! three prompts, so the `PQ2_0` ids are a valid oracle), plus the decoded
//! 32-token text against `Ternary-Bonsai-2-27B-PTQ1_0.prompt{i}.txt`.
//!
//! # Teacher forcing
//!
//! Each step feeds the **golden** token back in, while asserting that our
//! own argmax equals it. When every step matches — the pass condition —
//! this is exactly greedy decoding; when one does not, the remaining steps
//! are still measured against the right history instead of drifting, so a
//! failure report shows every divergent step rather than only the first.
//! Steps 24..32 (the text-only tail, which has no id oracle) feed our own
//! argmax.
//!
//! # Measured results (ruling R2' item 5)
//!
//! Real 27B on the 24 GB M3, CPU path, `f16` KV, teacher-forced as above
//! (the run re-prints all of this: one table row per step and a `LOCALISE`
//! line per prompt):
//!
//! | prompt | G1 ids | top-10 set / order | worst \|Δlogprob\| vs fork Metal (step, rank, id) | worst vs fork CPU | fork Metal-vs-CPU, worst |
//! |---|---|---|---|---|---|
//! | 1 | 24/24 | 24/24 same set, 24/24 same order | 2.22e-3 (19, 9, 5328) | 6.37e-2 | 6.30e-2 |
//! | 2 | 24/24 | 24/24 same set, 24/24 same order | 2.19e-3 (1, 6, 2) | 6.02e-2 | 5.96e-2 |
//! | 3 | 24/24 | 24/24 same set, 24/24 same order | 1.39e-2 (23, 9, 424) | 1.94e-1 | 2.08e-1 |
//!
//! All three 32-token continuations are byte-identical to the fork's text
//! dumps, and `PTQ1_0` reproduces every `PQ2_0` logprob bit for bit (so its
//! G1 and text results are the same table). Against the fork's **CPU**
//! backend our worst deltas track the fork's own Metal-vs-CPU spread
//! step by step: we sit next to its Metal backend, which is the oracle the
//! ruling names.
//!
//! **The prompt-3 outlier, localised.** Steps 0–22 of prompt 3 stay at or
//! below 2.25e-3 like the other prompts; the 1.39e-2 is one value, at
//! step 23 (the last golden step of the longest context, 36 positions),
//! golden rank 9 (token 424, logprob ≈ −5.77). At that same step the fork's
//! *own* two backends move apart by 2.08e-1 over the same top-10 (up from
//! ≤ 5.7e-2 at every earlier step), and ours lies between them at every
//! rank — 1.39e-2 from Metal, 1.94e-1 from CPU (e.g. token 424: Metal
//! −5.768, ours −5.782, CPU −5.976). The step is numerically sensitive for
//! every implementation, the deviation sits in the low-probability tail,
//! and the top-1/top-2 gap there (2.17) is nowhere near a greedy flip.

use std::time::Instant;

use oxibonsai_core::gguf::reader::{mmap_gguf_file, GgufFile};
use oxibonsai_model::hybrid::model::HybridModel;

use crate::harness::{
    argmax, compare_step, detokenize, gguf_vocab, golden_cpu_dir, golden_dir, gpt2_byte_decoder,
    locate_model, log_softmax, parse_golden_steps, parse_prompt_tokens, read_golden,
    read_golden_bytes, real_model_serial, record_capability, show_bytes, top_n, GoldenStep,
    StepComparison, G4_LOGPROB_BAND, PQ2_ENV, PQ2_FILE, PROMPTS, PTQ1_ENV, PTQ1_FILE, TOP_N,
};

/// KV window for the real-model runs: generous over the longest prompt
/// (13 tokens) plus 32 generated tokens, and only 256 MiB of `f16` KV even
/// if it were all allocated.
const MAX_SEQ: usize = 4096;

/// Tokens decoded per prompt: the length of the fork's `llama-cli` text
/// dump (`-n 32`). The server goldens carry the first 24 of them.
const TEXT_STEPS: usize = 32;

/// The tool appends this to every `llama-cli` continuation dump.
const TEXT_TRAILER: &[u8] = b"\n\n";

/// One prompt's run: our top-N rows per step plus the comparisons.
struct PromptRun {
    produced: Vec<u32>,
    vs_metal: Vec<StepComparison>,
    vs_cpu: Vec<Option<StepComparison>>,
    fork_spread: Vec<Option<f64>>,
    text: Vec<u8>,
}

/// Worst `|Δlogprob|` between two goldens' common top-N (the fork's own
/// backend spread at one step).
fn golden_spread(a: &GoldenStep, b: &GoldenStep) -> f64 {
    let mut worst = 0.0f64;
    for &(id, lp) in &a.top {
        if let Some(&(_, other)) = b.top.iter().find(|p| p.0 == id) {
            worst = worst.max((lp - other).abs());
        }
    }
    worst
}

/// Decode one prompt teacher-forced against `metal` for its steps, then
/// greedily to [`TEXT_STEPS`], printing one table row per step.
#[allow(clippy::too_many_arguments)]
fn run_prompt(
    label: &str,
    model: &mut HybridModel<'_>,
    prompt_index: usize,
    tokens: &[u32],
    metal: &[GoldenStep],
    cpu: &[GoldenStep],
    vocab: &[String],
) -> PromptRun {
    let decoder = gpt2_byte_decoder();
    let vocab_size = model.config().base.vocab_size;
    let mut logits = vec![0.0f32; vocab_size];
    model.reset();
    let t0 = Instant::now();
    model
        .forward_prefill(tokens, 0, &mut logits)
        .unwrap_or_else(|e| panic!("{label} prompt {prompt_index}: prefill failed: {e}"));
    eprintln!(
        "{label} prompt {prompt_index}: prefill {} tokens in {:.2}s",
        tokens.len(),
        t0.elapsed().as_secs_f64()
    );
    eprintln!(
        "{label} prompt {prompt_index}: step | golden ours | set order | worst|dlp| metal (rank,id) | \
         worst|dlp| cpu | fork metal-vs-cpu | golden gap"
    );
    let mut run = PromptRun {
        produced: Vec::with_capacity(TEXT_STEPS),
        vs_metal: Vec::with_capacity(metal.len()),
        vs_cpu: Vec::with_capacity(metal.len()),
        fork_spread: Vec::with_capacity(metal.len()),
        text: Vec::new(),
    };
    let decode_t0 = Instant::now();
    for step in 0..TEXT_STEPS {
        let ours = u32::try_from(argmax(&logits)).expect("token id fits u32");
        let feed = if let Some(golden) = metal.get(step) {
            let logprobs = log_softmax(&logits);
            let cmp = compare_step(&logprobs, ours, golden);
            let cmp_cpu = cpu.get(step).map(|g| compare_step(&logprobs, ours, g));
            let spread = cpu.get(step).map(|g| golden_spread(golden, g));
            let ours_top = top_n(&logprobs, TOP_N);
            eprintln!(
                "{label} p{prompt_index} {step:>4} | {:>6} {:>6} | {:>3} {:>5} | {:>10.3e} ({},{}) | {:>10} | {:>10} | {:.4}",
                cmp.golden_id,
                cmp.ours_id,
                if cmp.same_set { "ok" } else { "SET" },
                if cmp.same_order { "same" } else { "diff" },
                cmp.worst_delta,
                cmp.worst_at.0,
                cmp.worst_at.1,
                cmp_cpu
                    .as_ref()
                    .map_or_else(|| "-".to_string(), |c| format!("{:.3e}", c.worst_delta)),
                spread.map_or_else(|| "-".to_string(), |s| format!("{s:.3e}")),
                cmp.golden_gap,
            );
            // Full-precision baseline row: diffable across builds.
            let row: Vec<String> = ours_top
                .iter()
                .map(|(id, lp)| format!("{id}:{lp:.17e}"))
                .collect();
            eprintln!("BASELINE {label} p{prompt_index} s{step} {}", row.join(" "));
            run.vs_metal.push(cmp);
            run.vs_cpu.push(cmp_cpu);
            run.fork_spread.push(spread);
            golden.id
        } else {
            ours
        };
        run.produced.push(ours);
        let pos = tokens.len() + step;
        model
            .forward(feed, pos, &mut logits)
            .unwrap_or_else(|e| panic!("{label} prompt {prompt_index}: decode at {pos}: {e}"));
    }
    let elapsed = decode_t0.elapsed().as_secs_f64();
    eprintln!(
        "{label} prompt {prompt_index}: {TEXT_STEPS} decode steps in {elapsed:.2}s ({:.3} tok/s)",
        TEXT_STEPS as f64 / elapsed.max(1e-9)
    );
    run.text = detokenize(vocab, &decoder, &run.produced);
    run
}

/// The worst-step localisation line (ruling R2' item 5).
fn localise(label: &str, prompt_index: usize, run: &PromptRun) -> Option<(usize, f64)> {
    let (step, cmp) = run.vs_metal.iter().enumerate().max_by(|a, b| {
        a.1.worst_delta
            .partial_cmp(&b.1.worst_delta)
            .unwrap_or(std::cmp::Ordering::Equal)
    })?;
    let cpu = run
        .vs_cpu
        .get(step)
        .and_then(Option::as_ref)
        .map(|c| c.worst_delta);
    let spread = run.fork_spread.get(step).copied().flatten();
    eprintln!(
        "LOCALISE {label} prompt {prompt_index}: worst |dlogprob| vs fork-Metal = {:.4e} at step \
         {step} (golden rank {}, token {}); ours vs fork-CPU there = {}; the fork's own \
         Metal-vs-CPU spread there = {}; golden top-1/top-2 gap there = {:.4}",
        cmp.worst_delta,
        cmp.worst_at.0,
        cmp.worst_at.1,
        cpu.map_or_else(|| "-".to_string(), |c| format!("{c:.4e}")),
        spread.map_or_else(|| "-".to_string(), |s| format!("{s:.4e}")),
        cmp.golden_gap,
    );
    Some((step, cmp.worst_delta))
}

/// Load one prompt's `llama-cli` goldens for the build named by
/// `file_prefix`: `(prompt token ids, 32-token continuation text bytes)`.
/// The per-step server goldens are loaded separately
/// ([`metal_steps`] / [`cpu_steps`]).
fn load_goldens(file_prefix: &str, prompt_index: usize) -> (Vec<u32>, Vec<u8>) {
    let golden = golden_dir();
    let tokens = parse_prompt_tokens(&read_golden(
        &golden,
        &format!("{file_prefix}.prompt{prompt_index}.prompt_tokens.txt"),
    ));
    assert!(
        !tokens.is_empty(),
        "{file_prefix} prompt {prompt_index}: no tokens"
    );
    let text = read_golden_bytes(&golden, &format!("{file_prefix}.prompt{prompt_index}.txt"));
    (tokens, text)
}

fn metal_steps(prompt_index: usize) -> Vec<GoldenStep> {
    let steps = parse_golden_steps(&read_golden(
        &golden_dir(),
        &format!("PQ2_0.prompt{prompt_index}.server.json"),
    ));
    assert_eq!(steps.len(), 24, "prompt {prompt_index}: 24 golden steps");
    steps
}

fn cpu_steps(prompt_index: usize) -> Vec<GoldenStep> {
    let dir = golden_cpu_dir();
    let name = format!("PQ2_0.prompt{prompt_index}.server.json");
    if dir.join(&name).is_file() {
        parse_golden_steps(&read_golden(&dir, &name))
    } else {
        eprintln!(
            "note: no fork-CPU golden at {} — the informational CPU table is skipped",
            dir.join(&name).display()
        );
        Vec::new()
    }
}

/// Check the decoded continuation against the fork's text dump.
fn check_text(
    label: &str,
    prompt_index: usize,
    run: &PromptRun,
    golden_text: &[u8],
) -> Option<String> {
    let mut expected_body = golden_text.to_vec();
    if expected_body.ends_with(TEXT_TRAILER) {
        expected_body.truncate(expected_body.len() - TEXT_TRAILER.len());
    }
    if run.text == expected_body {
        eprintln!(
            "{label} prompt {prompt_index}: text OK: \"{}\"",
            show_bytes(&run.text)
        );
        None
    } else {
        Some(format!(
            "{label} prompt {prompt_index}: decoded continuation differs from the fork's text\n  \
             ours:   \"{}\"\n  golden: \"{}\"",
            show_bytes(&run.text),
            show_bytes(&expected_body)
        ))
    }
}

/// Collect the G1/G4 failures of one prompt (every violation, not the first).
fn g1_g4_failures(
    label: &str,
    prompt_index: usize,
    run: &PromptRun,
    check_g4: bool,
) -> Vec<String> {
    let mut failures = Vec::new();
    for (step, cmp) in run.vs_metal.iter().enumerate() {
        if cmp.ours_id != cmp.golden_id {
            failures.push(format!(
                "{label} prompt {prompt_index} step {step}: G1 greedy token {} != fork {} \
                 (golden gap {:.4})",
                cmp.ours_id, cmp.golden_id, cmp.golden_gap
            ));
        }
        if check_g4 && !cmp.same_set {
            failures.push(format!(
                "{label} prompt {prompt_index} step {step}: G4 top-{TOP_N} id set differs \
                 (fork-only {:?}, ours-only {:?})",
                cmp.missing, cmp.extra
            ));
        }
        if check_g4 && cmp.worst_delta > G4_LOGPROB_BAND {
            failures.push(format!(
                "{label} prompt {prompt_index} step {step}: G4 worst |dlogprob| {:.4e} > \
                 {G4_LOGPROB_BAND:e} at golden rank {} (token {})",
                cmp.worst_delta, cmp.worst_at.0, cmp.worst_at.1
            ));
        }
    }
    failures
}

/// G1 + G4 (ruling R2') + the 32-token text on the real `PQ2_0` 27B.
#[test]
fn hybrid_real_27b_pq2_0_matches_the_fork_goldens_bonsai2() {
    const TEST: &str = "oxibonsai-model::hybrid_forward_parity_tests::\
                        hybrid_real_27b_pq2_0_matches_the_fork_goldens_bonsai2";
    let Some(path) = locate_model(PQ2_ENV, PQ2_FILE, TEST) else {
        return;
    };
    let _one_real_model_at_a_time = real_model_serial();
    let mmap = mmap_gguf_file(&path).unwrap_or_else(|e| panic!("mmap {}: {e}", path.display()));
    let gguf = GgufFile::parse(&mmap).expect("27B PQ2_0 GGUF parses");
    let load = Instant::now();
    let mut model = HybridModel::from_gguf(&gguf, MAX_SEQ).expect("27B PQ2_0 loads");
    eprintln!("PQ2_0: loaded in {:.2}s", load.elapsed().as_secs_f64());
    assert_eq!(model.config().base.num_layers, 64);
    assert_eq!(model.split().full_layers().len(), 16);
    assert!(
        model.kv_is_f16(),
        "the fork's goldens were produced with an f16 KV cache (no -ctk/-ctv)"
    );
    let vocab = gguf_vocab(&gguf);
    let decoder = gpt2_byte_decoder();

    let mut failures = Vec::new();
    for prompt_index in 1..=3usize {
        let (tokens, text) = load_goldens("Ternary-Bonsai-2-27B-PQ2_0", prompt_index);
        // The fork's tokenisation of the raw prompt decodes back to it.
        let prompt_text = detokenize(&vocab, &decoder, &tokens);
        assert_eq!(
            prompt_text,
            PROMPTS[prompt_index - 1].as_bytes(),
            "prompt {prompt_index}: golden prompt tokens must decode to the raw prompt"
        );
        let metal = metal_steps(prompt_index);
        let cpu = cpu_steps(prompt_index);
        let run = run_prompt(
            "PQ2_0",
            &mut model,
            prompt_index,
            &tokens,
            &metal,
            &cpu,
            &vocab,
        );
        let _ = localise("PQ2_0", prompt_index, &run);
        let ids: Vec<u32> = metal.iter().map(|s| s.id).collect();
        let order_mismatches = run.vs_metal.iter().filter(|c| !c.same_order).count();
        eprintln!(
            "PQ2_0 prompt {prompt_index}: G1 {}/24 ids equal; top-{TOP_N} order differs at \
             {order_mismatches}/24 steps (recorded, not asserted)",
            run.produced
                .iter()
                .zip(&ids)
                .filter(|(a, b)| a == b)
                .count()
        );
        failures.extend(g1_g4_failures("PQ2_0", prompt_index, &run, true));
        failures.extend(check_text("PQ2_0", prompt_index, &run, &text));
    }
    assert!(
        failures.is_empty(),
        "real PQ2_0 27B vs the fork goldens:\n{}",
        failures.join("\n")
    );
    record_capability(true, TEST);
}

/// G1 on the real `PTQ1_0` 27B against the `PQ2_0` greedy ids, plus its
/// own 32-token text dump.
#[test]
fn hybrid_real_27b_ptq1_0_greedy_matches_the_fork_goldens_bonsai2() {
    const TEST: &str = "oxibonsai-model::hybrid_forward_parity_tests::\
                        hybrid_real_27b_ptq1_0_greedy_matches_the_fork_goldens_bonsai2";
    let Some(path) = locate_model(PTQ1_ENV, PTQ1_FILE, TEST) else {
        return;
    };
    let _one_real_model_at_a_time = real_model_serial();
    let mmap = mmap_gguf_file(&path).unwrap_or_else(|e| panic!("mmap {}: {e}", path.display()));
    let gguf = GgufFile::parse(&mmap).expect("27B PTQ1_0 GGUF parses");
    let load = Instant::now();
    let mut model = HybridModel::from_gguf(&gguf, MAX_SEQ).expect("27B PTQ1_0 loads");
    eprintln!("PTQ1_0: loaded in {:.2}s", load.elapsed().as_secs_f64());
    assert!(model.kv_is_f16());
    let vocab = gguf_vocab(&gguf);

    let mut failures = Vec::new();
    for prompt_index in 1..=3usize {
        let (tokens, text) = load_goldens("Ternary-Bonsai-2-27B-PTQ1_0", prompt_index);
        let (pq2_tokens, _) = load_goldens("Ternary-Bonsai-2-27B-PQ2_0", prompt_index);
        assert_eq!(
            tokens, pq2_tokens,
            "prompt {prompt_index}: the fork tokenised the prompt identically for both bands"
        );
        let metal = metal_steps(prompt_index);
        let cpu = cpu_steps(prompt_index);
        let run = run_prompt(
            "PTQ1_0",
            &mut model,
            prompt_index,
            &tokens,
            &metal,
            &cpu,
            &vocab,
        );
        let _ = localise("PTQ1_0", prompt_index, &run);
        // G1 only: the PQ2_0 logprobs are not a PTQ1_0 G4 oracle (the
        // ruling defines G4 on PQ2_0), so the logprob columns above are
        // informational here.
        failures.extend(g1_g4_failures("PTQ1_0", prompt_index, &run, false));
        failures.extend(check_text("PTQ1_0", prompt_index, &run, &text));
    }
    assert!(
        failures.is_empty(),
        "real PTQ1_0 27B vs the fork goldens:\n{}",
        failures.join("\n")
    );
    record_capability(true, TEST);
}
