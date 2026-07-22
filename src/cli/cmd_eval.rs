//! `oxibonsai eval` — evaluate a model's generation quality (ROUGE-1/2/L)
//! against a JSONL dataset. Requires the `eval` build feature.

use super::util::{missing_tokenizer_warning, resolve_tokenizer};

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    model: Option<String>,
    dataset: String,
    limit: Option<usize>,
    max_tokens: usize,
    max_seq_len: usize,
    tokenizer: Option<String>,
    report_json: Option<String>,
    report_markdown: Option<String>,
) -> anyhow::Result<()> {
    // Load + validate the dataset *before* touching the (possibly
    // large) model file, so a bad --dataset path fails fast with a
    // clear message instead of after paying the model-load cost.
    let dataset_text = std::fs::read_to_string(&dataset)
        .map_err(|e| anyhow::anyhow!("failed to read dataset '{dataset}': {e}"))?;
    let eval_set = oxibonsai_eval::EvalDataset::from_jsonl("eval", &dataset_text)
        .map_err(|e| anyhow::anyhow!("failed to parse dataset '{dataset}': {e}"))?;
    if eval_set.is_empty() {
        anyhow::bail!("dataset '{dataset}' contains no examples");
    }
    let examples: Vec<&oxibonsai_eval::EvalExample> = match limit {
        Some(n) => eval_set.examples.iter().take(n).collect(),
        None => eval_set.examples.iter().collect(),
    };

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

    let params = oxibonsai_runtime::sampling::SamplingParams {
        temperature: 0.0,
        ..oxibonsai_runtime::sampling::SamplingParams::default()
    };
    let mut engine = oxibonsai_runtime::InferenceEngine::from_gguf(&gguf, params, 42, max_seq_len)?;

    let lookup = resolve_tokenizer(tokenizer.as_deref(), &model);
    let tok = match &lookup.found {
        Some(p) => oxibonsai_runtime::TokenizerBridge::from_file(p)?,
        None => anyhow::bail!(
            "eval requires a tokenizer to encode dataset prompts, but none was found: {}",
            missing_tokenizer_warning(&lookup.searched)
        ),
    };

    tracing::info!(
        model = %model,
        dataset = %dataset,
        n_examples = examples.len(),
        "starting eval"
    );

    let mut pairs: Vec<(String, String)> = Vec::with_capacity(examples.len());
    for ex in &examples {
        let reference = ex.expected_output.clone().unwrap_or_default();
        let prompt_tokens = tok.encode(&ex.input)?;
        // Reset the KV cache so every example starts from a clean
        // position 0 (this engine is single-use for eval, not a
        // multi-turn session).
        engine.reset();
        let out_tokens = engine.generate(&prompt_tokens, max_tokens)?;
        let generated = tok.decode(&out_tokens)?;
        pairs.push((generated, reference));
    }

    let pair_refs: Vec<(&str, &str)> = pairs
        .iter()
        .map(|(g, r)| (g.as_str(), r.as_str()))
        .collect();
    let rouge = oxibonsai_eval::CorpusRouge::compute(&pair_refs);

    println!("{}", rouge.summary());

    let mut report = oxibonsai_eval::EvalReport::new(&model);
    if let Some(r1) = &rouge.rouge_1 {
        report.add(oxibonsai_eval::EvalResultEntry {
            task: dataset.clone(),
            metric: "rouge-1_f1".to_string(),
            value: r1.f1,
            unit: "F1".to_string(),
            notes: Some(format!("n={}", rouge.num_samples)),
        });
    }
    if let Some(r2) = &rouge.rouge_2 {
        report.add(oxibonsai_eval::EvalResultEntry {
            task: dataset.clone(),
            metric: "rouge-2_f1".to_string(),
            value: r2.f1,
            unit: "F1".to_string(),
            notes: Some(format!("n={}", rouge.num_samples)),
        });
    }
    if let Some(rl) = &rouge.rouge_l {
        report.add(oxibonsai_eval::EvalResultEntry {
            task: dataset.clone(),
            metric: "rouge-l_f1".to_string(),
            value: rl.f1,
            unit: "F1".to_string(),
            notes: Some(format!("n={}", rouge.num_samples)),
        });
    }

    if let Some(path) = &report_json {
        std::fs::write(path, report.to_json())
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
