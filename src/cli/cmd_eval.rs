//! `oxibonsai eval` — evaluate a model against a real evaluation task,
//! wired to the loaded engine end to end (RAG-EVAL-IMG-16).
//!
//! Before this, only ROUGE had a path from a real model; the other 15
//! evaluators in `oxibonsai-eval` were reachable only from that crate's
//! own unit tests. `--task` selects one of two scoring strategies:
//!
//! * **Generation-based** (rouge, bleu, chrf, meteor, qa, gsm8k,
//!   exact-match, perplexity): the model actually generates (or, for
//!   perplexity, is teacher-forced over) each example's `input`, exactly
//!   as `run`/`chat` would.
//! * **Logit-based** (mmlu, arc-easy, arc-challenge, hellaswag,
//!   winogrande, boolq, truthfulqa-mc1/mc2, calibration): the
//!   generation-free multiple-choice protocol these benchmarks actually
//!   specify — [`score_choices_logprob`] teacher-forces every candidate
//!   continuation and sums its per-token log-probability, reusing
//!   [`oxibonsai_runtime::InferenceEngine::prefill_from_pos`] /
//!   `decode_step` / `rewind_cache` (the same trio the speculative
//!   decoder uses) so the shared context is prefilled once per question
//!   and only rewound between candidates, and
//!   [`oxibonsai_runtime::api_types::compute_logprobs`] for the
//!   log-softmax rather than a fourth reimplementation of it.

use std::path::Path;

use oxibonsai_eval::{
    ArcEvaluator, ArcResult, ArcSplit, BoolQDataset, BoolQEvaluator, BoolQItem, EvalDataset,
    EvalReport, EvalResultEntry, ExactMatchEvaluator, Gsm8kEvaluator, HellaSwagDataset,
    HellaSwagEvaluator, McDataset, MmluEvaluator, PerplexityEvaluator, TruthfulQaDataset,
    TruthfulQaEvaluator, TruthfulQaMode, WinoGrandeDataset, WinoGrandeEvaluator,
};
use oxibonsai_runtime::{InferenceEngine, TokenizerBridge};

use super::args::EvalTask;
use super::util::{
    build_sampling_params, check_tokenizer_model_compatibility, missing_tokenizer_warning,
    model_vocab_size, resolve_tokenizer_vocab_aware,
};

/// Resolved arguments for `oxibonsai eval`, merged from CLI flags and
/// `--config` in `mod.rs` (cli-04).
pub(crate) struct EvalArgs {
    pub(crate) model: Option<String>,
    pub(crate) task: EvalTask,
    pub(crate) dataset: String,
    pub(crate) limit: Option<usize>,
    pub(crate) max_tokens: usize,
    pub(crate) max_seq_len: usize,
    pub(crate) tokenizer: Option<String>,
    pub(crate) report_json: Option<String>,
    pub(crate) report_markdown: Option<String>,
    pub(crate) allow_vocab_mismatch: bool,
}

/// The dataset, already loaded and `--limit`-truncated in the shape
/// `--task` requires, before any (possibly large) model is opened.
enum LoadedDataset {
    Generation(EvalDataset),
    Mc(McDataset),
    HellaSwag(HellaSwagDataset),
    WinoGrande(WinoGrandeDataset),
    BoolQ(BoolQDataset),
    TruthfulQa(TruthfulQaDataset),
}

pub(crate) fn run(args: EvalArgs) -> anyhow::Result<()> {
    let EvalArgs {
        model,
        task,
        dataset,
        limit,
        max_tokens,
        max_seq_len,
        tokenizer,
        report_json,
        report_markdown,
        allow_vocab_mismatch,
    } = args;

    // Load + validate the dataset in the shape `--task` expects *before*
    // touching the (possibly large) model file, so a bad --dataset path
    // fails fast — preserved for every task, not only the original
    // `rouge` default.
    let loaded = load_dataset_for_task(task, &dataset, limit)?;

    let model = model
        .or_else(|| std::env::var("OXI_MODEL").ok().filter(|s| !s.is_empty()))
        .ok_or_else(|| {
            anyhow::anyhow!("no model: pass --model <gguf> or set OXI_MODEL (e.g. in .env)")
        })?;
    let tokenizer = tokenizer.or_else(|| {
        std::env::var("OXI_TOKENIZER")
            .ok()
            .filter(|s| !s.is_empty())
    });

    let mmap = oxibonsai_core::gguf::reader::mmap_gguf_file(std::path::Path::new(&model))
        .map_err(|e| anyhow::anyhow!("failed to open model '{model}': {e}"))?;
    let gguf = oxibonsai_core::gguf::reader::GgufFile::parse(&mmap)?;

    // Eval is greedy-by-construction: the shared constructor (orchestrator
    // P0 addendum) with repetition_penalty=1.0 so it is exactly argmax,
    // matching every other backend's `--temperature 0` contract.
    let params = build_sampling_params(0.0, 40, 0.9, 1.0);
    let mut engine = InferenceEngine::from_gguf(&gguf, params, 42, max_seq_len)?;

    // TOK-08: vocab-aware resolution + a hard compatibility check.
    let expected_vocab = model_vocab_size(&gguf).ok();
    let lookup = resolve_tokenizer_vocab_aware(tokenizer.as_deref(), &model, expected_vocab);
    let tok = match &lookup.found {
        Some(p) => {
            let tok = TokenizerBridge::from_file(p)?;
            check_tokenizer_model_compatibility(&tok, p, &gguf, allow_vocab_mismatch)?;
            tok
        }
        None => anyhow::bail!(
            "eval requires a tokenizer to encode dataset prompts, but none was found: {}",
            missing_tokenizer_warning(&lookup.searched)
        ),
    };

    tracing::info!(model = %model, dataset = %dataset, task = ?task, "starting eval");

    let mut report = EvalReport::new(&model);
    run_task(
        task,
        &mut engine,
        &tok,
        loaded,
        max_tokens,
        &dataset,
        &mut report,
    )?;

    if let Some(path) = &report_json {
        let json = report
            .try_to_json()
            .map_err(|e| anyhow::anyhow!("failed to serialise report: {e}"))?;
        std::fs::write(path, json)
            .map_err(|e| anyhow::anyhow!("failed to write report '{path}': {e}"))?;
        println!("Wrote JSON report to {path}");
    }
    if let Some(path) = &report_markdown {
        std::fs::write(path, report.to_markdown())
            .map_err(|e| anyhow::anyhow!("failed to write report '{path}': {e}"))?;
        println!("Wrote Markdown report to {path}");
    }

    Ok(())
}

/// Load and `--limit`-truncate the dataset file in the shape `task`
/// expects. Kept in exact 1:1 correspondence with [`run_task`]'s match on
/// `(task, LoadedDataset)` — the two are the only place that pairing is
/// expressed, so a new `EvalTask` variant must extend both.
fn load_dataset_for_task(
    task: EvalTask,
    dataset_path: &str,
    limit: Option<usize>,
) -> anyhow::Result<LoadedDataset> {
    use EvalTask::*;
    match task {
        Rouge | Bleu | Chrf | Meteor | Qa | Gsm8k | ExactMatch | Perplexity => Ok(
            LoadedDataset::Generation(load_generation_dataset(dataset_path, limit)?),
        ),
        Mmlu | ArcEasy | ArcChallenge | Calibration => {
            let content = std::fs::read_to_string(dataset_path)
                .map_err(|e| anyhow::anyhow!("failed to read dataset '{dataset_path}': {e}"))?;
            let mut mc = McDataset::from_jsonl("eval", &content)
                .map_err(|e| anyhow::anyhow!("failed to parse dataset '{dataset_path}': {e}"))?;
            if let Some(n) = limit {
                mc.questions.truncate(n);
            }
            if mc.is_empty() {
                anyhow::bail!("dataset '{dataset_path}' contains no examples");
            }
            Ok(LoadedDataset::Mc(mc))
        }
        Hellaswag => {
            let mut ds = HellaSwagDataset::from_jsonl(Path::new(dataset_path))
                .map_err(|e| anyhow::anyhow!("failed to parse dataset '{dataset_path}': {e}"))?;
            if let Some(n) = limit {
                ds.items.truncate(n);
            }
            if ds.is_empty() {
                anyhow::bail!("dataset '{dataset_path}' contains no examples");
            }
            Ok(LoadedDataset::HellaSwag(ds))
        }
        Winogrande => {
            let mut ds = WinoGrandeDataset::from_jsonl(Path::new(dataset_path))
                .map_err(|e| anyhow::anyhow!("failed to parse dataset '{dataset_path}': {e}"))?;
            if let Some(n) = limit {
                ds.items.truncate(n);
            }
            if ds.is_empty() {
                anyhow::bail!("dataset '{dataset_path}' contains no examples");
            }
            Ok(LoadedDataset::WinoGrande(ds))
        }
        Boolq => {
            let mut ds = load_boolq_dataset(dataset_path)?;
            if let Some(n) = limit {
                ds.items.truncate(n);
            }
            if ds.is_empty() {
                anyhow::bail!("dataset '{dataset_path}' contains no examples");
            }
            Ok(LoadedDataset::BoolQ(ds))
        }
        TruthfulqaMc1 | TruthfulqaMc2 => {
            let mut ds = TruthfulQaDataset::from_jsonl(Path::new(dataset_path))
                .map_err(|e| anyhow::anyhow!("failed to parse dataset '{dataset_path}': {e}"))?;
            if let Some(n) = limit {
                ds.items.truncate(n);
            }
            if ds.is_empty() {
                anyhow::bail!("dataset '{dataset_path}' contains no examples");
            }
            Ok(LoadedDataset::TruthfulQa(ds))
        }
    }
}

/// `BoolQDataset` has no `from_jsonl` in `oxibonsai-eval` (that crate file
/// is not in this package's `owned_files`); its `BoolQItem` fields are
/// `pub`, so this reads the JSONL directly and builds the dataset through
/// the existing `from_items` constructor instead.
fn load_boolq_dataset(dataset_path: &str) -> anyhow::Result<BoolQDataset> {
    let content = std::fs::read_to_string(dataset_path)
        .map_err(|e| anyhow::anyhow!("failed to read dataset '{dataset_path}': {e}"))?;
    let mut items = Vec::new();
    for (line_no, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line)
            .map_err(|e| anyhow::anyhow!("dataset '{dataset_path}' line {}: {e}", line_no + 1))?;
        let passage = v
            .get("passage")
            .and_then(|x| x.as_str())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "dataset '{dataset_path}' line {}: missing or invalid \"passage\"",
                    line_no + 1
                )
            })?
            .to_string();
        let question = v
            .get("question")
            .and_then(|x| x.as_str())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "dataset '{dataset_path}' line {}: missing or invalid \"question\"",
                    line_no + 1
                )
            })?
            .to_string();
        let answer = v.get("answer").and_then(|x| x.as_bool()).ok_or_else(|| {
            anyhow::anyhow!(
                "dataset '{dataset_path}' line {}: missing or invalid \"answer\" (must be a \
                 JSON boolean)",
                line_no + 1
            )
        })?;
        items.push(BoolQItem {
            passage,
            question,
            answer,
        });
    }
    Ok(BoolQDataset::from_items(items))
}

/// Load a generic `{"input", "expected_output"}` JSONL dataset for a
/// generation-based task, truncated to `--limit` examples.
fn load_generation_dataset(
    dataset_path: &str,
    limit: Option<usize>,
) -> anyhow::Result<EvalDataset> {
    let dataset_text = std::fs::read_to_string(dataset_path)
        .map_err(|e| anyhow::anyhow!("failed to read dataset '{dataset_path}': {e}"))?;
    let eval_set = EvalDataset::from_jsonl("eval", &dataset_text)
        .map_err(|e| anyhow::anyhow!("failed to parse dataset '{dataset_path}': {e}"))?;
    if eval_set.is_empty() {
        anyhow::bail!("dataset '{dataset_path}' contains no examples");
    }
    let limited = match limit {
        Some(n) => {
            let mut d = EvalDataset::new(&eval_set.name);
            for ex in eval_set.examples.into_iter().take(n) {
                d.add(ex);
            }
            d
        }
        None => eval_set,
    };
    Ok(limited)
}

/// Generate a real completion for every example's `input`, resetting the
/// engine's KV cache between examples (eval is single-use per example,
/// not a multi-turn session).
fn generate_completions(
    engine: &mut InferenceEngine<'_>,
    tok: &TokenizerBridge,
    dataset: &EvalDataset,
    max_tokens: usize,
) -> anyhow::Result<Vec<String>> {
    let mut completions = Vec::with_capacity(dataset.examples.len());
    for ex in &dataset.examples {
        let prompt_tokens = tok.encode(&ex.input)?;
        engine.reset();
        let out_tokens = engine.generate(&prompt_tokens, max_tokens)?;
        completions.push(tok.decode(&out_tokens)?);
    }
    Ok(completions)
}

/// Score every choice of every question in `mc` via
/// [`score_choices_logprob`].
fn score_mc_dataset(
    engine: &mut InferenceEngine<'_>,
    tok: &TokenizerBridge,
    mc: &McDataset,
) -> anyhow::Result<Vec<Vec<f32>>> {
    mc.questions
        .iter()
        .map(|q| score_choices_logprob(engine, tok, &q.question, &q.choices))
        .collect()
}

/// Score each of `choices` as a teacher-forced continuation of `context`,
/// returning the summed log-probability of each continuation's tokens
/// (the standard, generation-free multiple-choice protocol these
/// benchmarks specify).
///
/// `context` and each `choice` are encoded via the same joint-encode
/// technique `lm-evaluation-harness` uses (`Task._encode_pair`), not
/// encoded independently and concatenated: a tokenizer's BPE merges can
/// cross the context/continuation boundary (most commonly, a leading
/// space fuses into the next word's token), so
/// `encode(context) ++ encode(choice) != encode(context + choice)` in
/// general. Encoding each choice in isolation — the previous approach —
/// could silently run the context's last character into the choice's
/// first token (e.g. producing the single run-on token sequence for
/// `"...2+2?" + "4"` that a naive concatenation implies, instead of the
/// `"...2+2?" + " 4"` a real completion would need) whenever a dataset's
/// choices carry no leading space of their own, which is the common case
/// for HellaSwag/ARC/MMLU-style choice lists.
///
/// The technique: trim `context`'s own trailing whitespace before
/// encoding it ALONE (so its encoding ends on a boundary the tokenizer
/// would also produce mid-string, rather than an end-of-string-only
/// boundary), then jointly encode the *unmodified* `context + choice`
/// string and take each choice's continuation tokens as whatever lies
/// beyond the trimmed context's token count in that joint encoding.
///
/// The context is prefilled once (`prefill_from_pos`); each choice then
/// only replays its own continuation tokens via `decode_step` before
/// `rewind_cache` discards that choice's KV writes, so every choice after
/// the first reuses the same prefilled context instead of reprocessing it.
/// This is the same prefill → decode_step → rewind_cache trio the
/// speculative decoder uses, applied to scoring instead of drafting.
///
/// Deliberately NOT length-normalized: `lm-evaluation-harness` reports
/// `acc_norm` (dividing by each continuation's byte length) as the
/// headline HellaSwag metric, but the normalization convention differs
/// per task family (byte length vs token count vs no normalization at
/// all), and `HellaSwagEvaluator`/`ArcEvaluator`/etc.'s `evaluate_logits`
/// have no way to report a second, differently-named metric from one
/// call. This function's callers still report raw (un-normalized) `acc`,
/// same as before this fix.
fn score_choices_logprob(
    engine: &mut InferenceEngine<'_>,
    tok: &TokenizerBridge,
    context: &str,
    choices: &[String],
) -> anyhow::Result<Vec<f32>> {
    engine.reset();
    let trimmed_context = context.trim_end();
    let ctx_tokens = tok.encode(trimmed_context)?;
    if ctx_tokens.is_empty() {
        anyhow::bail!("context tokenized to zero tokens: {context:?}");
    }
    let ctx_len = ctx_tokens.len();
    let first_logits = engine.prefill_from_pos(&ctx_tokens, 0)?;

    let mut scores = Vec::with_capacity(choices.len());
    for choice in choices {
        // Joint encoding of the UNMODIFIED `context` (not `trimmed_context`)
        // plus `choice`: any whitespace trimmed off `context` above is
        // still present here, exactly where the real completion would
        // put it, so the tokenizer sees the identical boundary text a
        // real continuation would produce.
        let whole = format!("{context}{choice}");
        let whole_tokens = tok.encode(&whole)?;
        let cont_tokens: &[u32] = whole_tokens.get(ctx_len..).unwrap_or(&[]);
        if cont_tokens.is_empty() {
            scores.push(f32::NEG_INFINITY);
            continue;
        }
        let mut logprob_sum = 0.0f32;
        let mut cur_logits = first_logits.clone();
        for (i, &token_id) in cont_tokens.iter().enumerate() {
            let lp =
                oxibonsai_runtime::api_types::compute_logprobs(&cur_logits, token_id, 1, &|_| {
                    String::new()
                })
                .logprob;
            logprob_sum += lp;
            cur_logits = engine.decode_step(token_id, ctx_len + i)?;
        }
        scores.push(logprob_sum);
        engine.rewind_cache(ctx_len);
    }
    Ok(scores)
}

/// Numerically-stable softmax, used to turn [`score_choices_logprob`]'s
/// summed log-probabilities into a probability distribution for
/// [`oxibonsai_eval::calibration_all`].
fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return vec![0.0; logits.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

#[allow(clippy::too_many_lines)]
fn run_task(
    task: EvalTask,
    engine: &mut InferenceEngine<'_>,
    tok: &TokenizerBridge,
    loaded: LoadedDataset,
    max_tokens: usize,
    dataset_path: &str,
    report: &mut EvalReport,
) -> anyhow::Result<()> {
    match (task, loaded) {
        (EvalTask::Rouge, LoadedDataset::Generation(dataset)) => {
            let completions = generate_completions(engine, tok, &dataset, max_tokens)?;
            let pair_refs: Vec<(&str, &str)> = completions
                .iter()
                .zip(dataset.examples.iter())
                .map(|(gen, ex)| (gen.as_str(), ex.expected_output.as_deref().unwrap_or("")))
                .collect();
            let rouge = oxibonsai_eval::CorpusRouge::compute(&pair_refs);
            println!("{}", rouge.summary());
            if let Some(r1) = &rouge.rouge_1 {
                report.add(EvalResultEntry {
                    task: dataset_path.to_string(),
                    metric: "rouge-1_f1".to_string(),
                    value: r1.f1,
                    unit: "F1".to_string(),
                    notes: Some(format!("n={}", rouge.num_samples)),
                });
            }
            if let Some(r2) = &rouge.rouge_2 {
                report.add(EvalResultEntry {
                    task: dataset_path.to_string(),
                    metric: "rouge-2_f1".to_string(),
                    value: r2.f1,
                    unit: "F1".to_string(),
                    notes: Some(format!("n={}", rouge.num_samples)),
                });
            }
            if let Some(rl) = &rouge.rouge_l {
                report.add(EvalResultEntry {
                    task: dataset_path.to_string(),
                    metric: "rouge-l_f1".to_string(),
                    value: rl.f1,
                    unit: "F1".to_string(),
                    notes: Some(format!("n={}", rouge.num_samples)),
                });
            }
            Ok(())
        }

        (EvalTask::Bleu, LoadedDataset::Generation(dataset)) => {
            let completions = generate_completions(engine, tok, &dataset, max_tokens)?;
            let candidates: Vec<&str> = completions.iter().map(String::as_str).collect();
            let references: Vec<Vec<&str>> = dataset
                .examples
                .iter()
                .map(|ex| vec![ex.expected_output.as_deref().unwrap_or("")])
                .collect();
            let score = oxibonsai_eval::corpus_bleu(
                &candidates,
                &references,
                &oxibonsai_eval::BleuConfig::default(),
            );
            println!(
                "BLEU: {:.4} (brevity_penalty={:.3}, n={})",
                score.bleu,
                score.brevity_penalty,
                completions.len()
            );
            report.add(EvalResultEntry {
                task: dataset_path.to_string(),
                metric: "bleu".to_string(),
                value: score.bleu,
                unit: "score".to_string(),
                notes: Some(format!("n={}", completions.len())),
            });
            Ok(())
        }

        (EvalTask::Chrf, LoadedDataset::Generation(dataset)) => {
            let completions = generate_completions(engine, tok, &dataset, max_tokens)?;
            let mut total = 0.0f32;
            for (gen, ex) in completions.iter().zip(dataset.examples.iter()) {
                let reference = ex.expected_output.as_deref().unwrap_or("");
                total += oxibonsai_eval::chrf(gen, reference).score;
            }
            let n = completions.len();
            let mean = if n > 0 { total / n as f32 } else { 0.0 };
            println!("chrF: {mean:.4} (n={n})");
            report.add(EvalResultEntry {
                task: dataset_path.to_string(),
                metric: "chrf".to_string(),
                value: mean,
                unit: "score".to_string(),
                notes: Some(format!("n={n}")),
            });
            Ok(())
        }

        (EvalTask::Meteor, LoadedDataset::Generation(dataset)) => {
            let completions = generate_completions(engine, tok, &dataset, max_tokens)?;
            let cfg = oxibonsai_eval::MeteorConfig::default();
            let mut total = 0.0f32;
            for (gen, ex) in completions.iter().zip(dataset.examples.iter()) {
                let reference = ex.expected_output.as_deref().unwrap_or("");
                total += oxibonsai_eval::meteor(gen, reference, &cfg).score;
            }
            let n = completions.len();
            let mean = if n > 0 { total / n as f32 } else { 0.0 };
            println!("METEOR: {mean:.4} (n={n})");
            report.add(EvalResultEntry {
                task: dataset_path.to_string(),
                metric: "meteor".to_string(),
                value: mean,
                unit: "score".to_string(),
                notes: Some(format!("n={n}")),
            });
            Ok(())
        }

        (EvalTask::Qa, LoadedDataset::Generation(dataset)) => {
            let completions = generate_completions(engine, tok, &dataset, max_tokens)?;
            let examples: Vec<(String, Vec<String>)> = completions
                .iter()
                .zip(dataset.examples.iter())
                .map(|(gen, ex)| {
                    (
                        gen.clone(),
                        vec![ex.expected_output.clone().unwrap_or_default()],
                    )
                })
                .collect();
            let (em, f1) = oxibonsai_eval::corpus_em_f1(&examples);
            println!("QA: EM={:.4} F1={:.4} (n={})", em, f1, examples.len());
            report.add(EvalResultEntry {
                task: dataset_path.to_string(),
                metric: "qa_exact_match".to_string(),
                value: em,
                unit: "fraction".to_string(),
                notes: Some(format!("n={}", examples.len())),
            });
            report.add(EvalResultEntry {
                task: dataset_path.to_string(),
                metric: "qa_f1".to_string(),
                value: f1,
                unit: "fraction".to_string(),
                notes: Some(format!("n={}", examples.len())),
            });
            Ok(())
        }

        (EvalTask::Gsm8k, LoadedDataset::Generation(dataset)) => {
            let completions = generate_completions(engine, tok, &dataset, max_tokens)?;
            let result = Gsm8kEvaluator::new().evaluate_dataset(&dataset, &completions);
            println!(
                "GSM8K: {:.2}% ({}/{}), no_answer_rate={:.2}%",
                result.accuracy_pct(),
                result.correct,
                result.total,
                result.no_answer_rate() * 100.0
            );
            report.add(EvalResultEntry {
                task: dataset_path.to_string(),
                metric: "gsm8k_accuracy".to_string(),
                value: result.accuracy_pct(),
                unit: "%".to_string(),
                notes: Some(format!(
                    "n={}, no_answer_rate={:.2}%",
                    result.total,
                    result.no_answer_rate() * 100.0
                )),
            });
            Ok(())
        }

        (EvalTask::ExactMatch, LoadedDataset::Generation(dataset)) => {
            let completions = generate_completions(engine, tok, &dataset, max_tokens)?;
            let result = ExactMatchEvaluator::new().evaluate_dataset(&dataset, &completions);
            println!(
                "Exact Match: {:.2}% ({}/{})",
                result.accuracy_pct(),
                result.correct,
                result.total
            );
            report.add(EvalResultEntry {
                task: dataset_path.to_string(),
                metric: "exact_match".to_string(),
                value: result.accuracy_pct(),
                unit: "%".to_string(),
                notes: Some(format!("n={}", result.total)),
            });
            Ok(())
        }

        (EvalTask::Perplexity, LoadedDataset::Generation(dataset)) => {
            // Per-example log-probabilities, one `Vec<f32>` per example
            // (one entry per predicted token) — kept un-exponentiated so
            // they can be pooled corpus-wide before the single final
            // `exp()`, rather than only ever seeing each example's own
            // already-exponentiated PPL. Computed via the same
            // `compute_logprobs` log-softmax [`score_choices_logprob`]
            // uses, rather than a second reimplementation of it.
            let evaluator = PerplexityEvaluator::new();
            let mut log_probs_batch: Vec<Vec<f32>> = Vec::with_capacity(dataset.examples.len());
            for ex in &dataset.examples {
                let tokens = tok.encode(&ex.input)?;
                if tokens.len() < 2 {
                    // Nothing to predict from a 0- or 1-token sequence.
                    continue;
                }
                engine.reset();
                let mut log_probs: Vec<f32> = Vec::with_capacity(tokens.len() - 1);
                let mut cur = engine.decode_step(tokens[0], 0)?;
                for (i, &target) in tokens.iter().enumerate().skip(1) {
                    let lp =
                        oxibonsai_runtime::api_types::compute_logprobs(&cur, target, 1, &|_| {
                            String::new()
                        })
                        .logprob;
                    log_probs.push(lp);
                    cur = engine.decode_step(target, i)?;
                }
                log_probs_batch.push(log_probs);
            }
            if log_probs_batch.is_empty() {
                anyhow::bail!(
                    "no example in '{dataset_path}' tokenized to at least 2 tokens; cannot \
                     compute perplexity"
                );
            }

            // Headline metric: corpus perplexity `exp(Σ(-log p) / Σ
            // tokens)`, pooling every token across every example before
            // the single final `exp()` — the standard, publication-
            // comparable definition. The previous implementation reported
            // the arithmetic MEAN of each example's own already-exponentiated
            // PPL, a macro-average that over-weights short examples
            // relative to long ones and is not comparable to any published
            // PPL number, even though it was honestly labelled "mean".
            let corpus_ppl = evaluator.corpus_perplexity(&log_probs_batch);
            let per_example_ppls: Vec<f32> = log_probs_batch
                .iter()
                .map(|lp| evaluator.compute(lp))
                .collect();
            let n = per_example_ppls.len();
            let total_tokens: usize = log_probs_batch.iter().map(Vec::len).sum();
            let min_ppl = per_example_ppls
                .iter()
                .cloned()
                .fold(f32::INFINITY, f32::min);
            let max_ppl = per_example_ppls
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            println!(
                "Perplexity: corpus={corpus_ppl:.4} (exp(sum NLL / {total_tokens} tokens), \
                 n={n} examples) per-example[min={min_ppl:.4} max={max_ppl:.4}]"
            );
            report.add(EvalResultEntry {
                task: dataset_path.to_string(),
                metric: "perplexity".to_string(),
                value: corpus_ppl,
                unit: "PPL".to_string(),
                notes: Some(format!(
                    "corpus PPL over {total_tokens} tokens across n={n} examples; \
                     per-example min={min_ppl:.2} max={max_ppl:.2}"
                )),
            });
            Ok(())
        }

        (EvalTask::Mmlu, LoadedDataset::Mc(mc)) => {
            let per_choice_logits = score_mc_dataset(engine, tok, &mc)?;
            let result = MmluEvaluator::new().evaluate_logits(&mc, &per_choice_logits);
            println!(
                "MMLU: {:.2}% ({}/{})",
                result.accuracy_pct, result.correct, result.total
            );
            for (subject, acc) in &result.by_subject {
                println!(
                    "  {subject}: {:.2}% ({}/{})",
                    acc.accuracy_pct(),
                    acc.correct,
                    acc.total
                );
            }
            report.add(EvalResultEntry {
                task: dataset_path.to_string(),
                metric: "mmlu_accuracy".to_string(),
                value: result.accuracy_pct,
                unit: "%".to_string(),
                notes: Some(format!("n={}", result.total)),
            });
            Ok(())
        }

        (EvalTask::ArcEasy, LoadedDataset::Mc(mc)) => {
            let per_choice_logits = score_mc_dataset(engine, tok, &mc)?;
            let raw = ArcEvaluator::easy().evaluate_logits(&mc, &per_choice_logits);
            let result = ArcResult::from_accuracy_result(ArcSplit::Easy, raw);
            println!(
                "{}: {:.2}% ({}/{})",
                result.split_name(),
                result.accuracy_pct(),
                result.correct,
                result.total
            );
            report.add(EvalResultEntry {
                task: dataset_path.to_string(),
                metric: "arc_easy_accuracy".to_string(),
                value: result.accuracy_pct(),
                unit: "%".to_string(),
                notes: Some(format!("n={}", result.total)),
            });
            Ok(())
        }

        (EvalTask::ArcChallenge, LoadedDataset::Mc(mc)) => {
            let per_choice_logits = score_mc_dataset(engine, tok, &mc)?;
            let raw = ArcEvaluator::challenge().evaluate_logits(&mc, &per_choice_logits);
            let result = ArcResult::from_accuracy_result(ArcSplit::Challenge, raw);
            println!(
                "{}: {:.2}% ({}/{})",
                result.split_name(),
                result.accuracy_pct(),
                result.correct,
                result.total
            );
            report.add(EvalResultEntry {
                task: dataset_path.to_string(),
                metric: "arc_challenge_accuracy".to_string(),
                value: result.accuracy_pct(),
                unit: "%".to_string(),
                notes: Some(format!("n={}", result.total)),
            });
            Ok(())
        }

        (EvalTask::Calibration, LoadedDataset::Mc(mc)) => {
            let per_choice_logits = score_mc_dataset(engine, tok, &mc)?;
            let labels: Vec<usize> = mc.questions.iter().map(|q| q.correct_answer).collect();
            let probs: Vec<Vec<f32>> = per_choice_logits.iter().map(|l| softmax(l)).collect();
            let result = oxibonsai_eval::calibration_all(&probs, &per_choice_logits, &labels, 10)
                .map_err(|e| anyhow::anyhow!("calibration computation failed: {e}"))?;
            println!(
                "Calibration: ECE={:.4} Brier={:.4} NLL={:.4} (n={})",
                result.ece,
                result.brier,
                result.nll,
                labels.len()
            );
            report.add(EvalResultEntry {
                task: dataset_path.to_string(),
                metric: "calibration_ece".to_string(),
                value: result.ece,
                unit: "fraction".to_string(),
                notes: Some(format!("n={}", labels.len())),
            });
            report.add(EvalResultEntry {
                task: dataset_path.to_string(),
                metric: "calibration_brier".to_string(),
                value: result.brier,
                unit: "score".to_string(),
                notes: None,
            });
            report.add(EvalResultEntry {
                task: dataset_path.to_string(),
                metric: "calibration_nll".to_string(),
                value: result.nll,
                unit: "nats".to_string(),
                notes: None,
            });
            Ok(())
        }

        (EvalTask::Hellaswag, LoadedDataset::HellaSwag(ds)) => {
            let mut per_choice_logits = Vec::with_capacity(ds.items.len());
            for item in &ds.items {
                per_choice_logits.push(score_choices_logprob(
                    engine,
                    tok,
                    &item.ctx,
                    &item.endings,
                )?);
            }
            let result = HellaSwagEvaluator::new().evaluate_logits(&ds, &per_choice_logits);
            println!(
                "HellaSwag: {:.2}% ({}/{})",
                result.accuracy_pct, result.correct, result.total
            );
            report.add(EvalResultEntry {
                task: dataset_path.to_string(),
                metric: "hellaswag_accuracy".to_string(),
                value: result.accuracy_pct,
                unit: "%".to_string(),
                notes: Some(format!("n={}", result.total)),
            });
            Ok(())
        }

        (EvalTask::Winogrande, LoadedDataset::WinoGrande(ds)) => {
            let mut per_choice_logits = Vec::with_capacity(ds.items.len());
            for item in &ds.items {
                let choices = vec![item.option1.clone(), item.option2.clone()];
                per_choice_logits.push(score_choices_logprob(
                    engine,
                    tok,
                    &item.sentence,
                    &choices,
                )?);
            }
            let result = WinoGrandeEvaluator::new().evaluate_logits(&ds, &per_choice_logits);
            println!(
                "WinoGrande: {:.2}% ({}/{})",
                result.accuracy_pct(),
                result.correct,
                result.total
            );
            report.add(EvalResultEntry {
                task: dataset_path.to_string(),
                metric: "winogrande_accuracy".to_string(),
                value: result.accuracy_pct(),
                unit: "%".to_string(),
                notes: Some(format!("n={}", result.total)),
            });
            Ok(())
        }

        (EvalTask::Boolq, LoadedDataset::BoolQ(ds)) => {
            let evaluator = BoolQEvaluator::new();
            let choices = vec!["yes".to_string(), "no".to_string()];
            let mut logit_pairs = Vec::with_capacity(ds.items.len());
            for item in &ds.items {
                let prompt = evaluator.build_prompt(item);
                let scores = score_choices_logprob(engine, tok, &prompt, &choices)?;
                logit_pairs.push([scores[0], scores[1]]);
            }
            let result = evaluator.evaluate_logits(&ds, &logit_pairs);
            println!(
                "BoolQ: {:.2}% ({}/{}), predicted yes={} no={}",
                result.accuracy_pct,
                result.correct,
                result.total,
                result.yes_predicted,
                result.no_predicted
            );
            report.add(EvalResultEntry {
                task: dataset_path.to_string(),
                metric: "boolq_accuracy".to_string(),
                value: result.accuracy_pct,
                unit: "%".to_string(),
                notes: Some(format!("n={}", result.total)),
            });
            Ok(())
        }

        (EvalTask::TruthfulqaMc1, LoadedDataset::TruthfulQa(ds)) => run_truthfulqa(
            engine,
            tok,
            &ds,
            TruthfulQaEvaluator::mc1(),
            dataset_path,
            report,
        ),
        (EvalTask::TruthfulqaMc2, LoadedDataset::TruthfulQa(ds)) => run_truthfulqa(
            engine,
            tok,
            &ds,
            TruthfulQaEvaluator::mc2(),
            dataset_path,
            report,
        ),

        // Unreachable by construction: `load_dataset_for_task` and this
        // match are kept in exact 1:1 correspondence per `EvalTask`
        // variant (see both functions' doc comments).
        (task, _) => unreachable!(
            "load_dataset_for_task produced a LoadedDataset variant that does not match \
             {task:?}'s arm in run_task"
        ),
    }
}

fn run_truthfulqa(
    engine: &mut InferenceEngine<'_>,
    tok: &TokenizerBridge,
    ds: &TruthfulQaDataset,
    evaluator: TruthfulQaEvaluator,
    dataset_path: &str,
    report: &mut EvalReport,
) -> anyhow::Result<()> {
    let mut per_choice_logits = Vec::with_capacity(ds.items.len());
    for item in &ds.items {
        let choices = match evaluator.mode {
            TruthfulQaMode::Mc1 => &item.mc1_choices,
            TruthfulQaMode::Mc2 => &item.mc2_choices,
        };
        per_choice_logits.push(score_choices_logprob(engine, tok, &item.question, choices)?);
    }
    let mode_name = match evaluator.mode {
        TruthfulQaMode::Mc1 => "mc1",
        TruthfulQaMode::Mc2 => "mc2",
    };
    let result = evaluator.evaluate_logits(ds, &per_choice_logits);
    println!(
        "TruthfulQA {}: {:.2}% ({}/{}, skipped={})",
        mode_name.to_uppercase(),
        result.accuracy_pct,
        result.correct,
        result.total,
        result.skipped
    );
    report.add(EvalResultEntry {
        task: dataset_path.to_string(),
        metric: format!("truthfulqa_{mode_name}"),
        value: result.accuracy_pct,
        unit: "%".to_string(),
        notes: Some(format!("n={}, skipped={}", result.total, result.skipped)),
    });
    Ok(())
}
