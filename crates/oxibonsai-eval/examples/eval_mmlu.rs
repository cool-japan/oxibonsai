//! MMLU-style multiple-choice evaluation with `oxibonsai-eval`, end to end.
//!
//! The example loads a JSONL dataset in the MMLU shape (one object per line:
//! `{"id", "question", "choices": [...], "correct_answer", "subject"?}`),
//! scores every question, and prints the overall accuracy, the per-subject
//! breakdown and the Markdown report the harness produces.
//!
//! Without arguments it evaluates a small inline dataset, so it runs anywhere:
//!
//! ```bash
//! cargo run -p oxibonsai-eval --example eval_mmlu
//! cargo run -p oxibonsai-eval --example eval_mmlu -- path/to/mmlu.jsonl
//! ```
//!
//! The per-choice scores come from `stand_in_scorer`, a deterministic
//! word-overlap heuristic that keeps the example self-contained. A real run
//! feeds [`MmluEvaluator::evaluate_logits`] one log-probability per answer
//! option from a loaded model; `oxibonsai eval --task mmlu --model <model.gguf>
//! --dataset <mmlu.jsonl>` does exactly that against the real engine.

use std::collections::HashSet;

use oxibonsai_eval::accuracy::AccuracyResult;
use oxibonsai_eval::dataset::{McDataset, MultipleChoiceQuestion};
use oxibonsai_eval::error::EvalError;
use oxibonsai_eval::mmlu::MmluEvaluator;
use oxibonsai_eval::report::EvalReport;

/// A small inline dataset in the MMLU JSONL shape.
const SAMPLE: &str = r#"{"id":"astronomy/0","question":"Which planet is closest to the Sun?","choices":["Venus","Mercury","Mars","Jupiter"],"correct_answer":1,"subject":"astronomy"}
{"id":"astronomy/1","question":"A star forms from a collapsing cloud of gas and dust","choices":["Mostly a cloud of ice and rock","A collapsing cloud of gas and dust under gravity","A planet losing its atmosphere","A comet falling into the Sun"],"correct_answer":1,"subject":"astronomy"}
{"id":"biology/0","question":"What do plants absorb from the air during photosynthesis?","choices":["Oxygen released by animals","Nitrogen from the soil","Carbon dioxide absorbed from the air","Methane from the oceans"],"correct_answer":2,"subject":"biology"}
{"id":"biology/1","question":"Which organelle produces most of a cell's ATP?","choices":["The nucleus","The ribosome","The Golgi body","The mitochondrion"],"correct_answer":3,"subject":"biology"}
{"id":"computer_science/0","question":"Which data structure gives first in, first out access?","choices":["A queue gives first in, first out access","A stack","A binary tree","A hash map"],"correct_answer":0,"subject":"computer_science"}
{"id":"computer_science/1","question":"What does a compiler translate source code into?","choices":["Compiled spreadsheets","Machine code or an intermediate representation of the source code","Plain English","Network packets"],"correct_answer":1,"subject":"computer_science"}
"#;

/// Lower-cased alphanumeric words of `text`.
fn words(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// One score per answer option: how many words the option shares with the
/// question. The evaluator takes the argmax, so any monotone score works; a
/// real run supplies the model's log-probability of each option instead.
fn stand_in_scorer(question: &MultipleChoiceQuestion) -> Vec<f32> {
    let stem = words(&question.question);
    question
        .choices
        .iter()
        .map(|choice| words(choice).intersection(&stem).count() as f32)
        .collect()
}

fn main() -> Result<(), EvalError> {
    let (name, jsonl) = match std::env::args().nth(1) {
        Some(path) => {
            let text = std::fs::read_to_string(&path)?;
            (path, text)
        }
        None => ("inline-sample".to_string(), SAMPLE.to_string()),
    };

    let dataset = McDataset::from_jsonl(&name, &jsonl)?;
    if dataset.is_empty() {
        return Err(EvalError::DatasetEmpty);
    }

    let scores: Vec<Vec<f32>> = dataset.questions.iter().map(stand_in_scorer).collect();
    let result = MmluEvaluator::new().evaluate_logits(&dataset, &scores);

    println!(
        "MMLU on {name}: {}/{} correct = {:.1}%",
        result.correct, result.total, result.accuracy_pct
    );
    let mut subjects: Vec<_> = result.by_subject.iter().collect();
    subjects.sort_by(|a, b| a.0.cmp(b.0));
    for (subject, accuracy) in subjects {
        println!(
            "  {subject:<20} {}/{} = {:.1}%",
            accuracy.correct,
            accuracy.total,
            accuracy.accuracy * 100.0
        );
    }

    let mut report = EvalReport::new("stand-in-scorer");
    report.add_accuracy(
        "mmlu",
        &AccuracyResult {
            correct: result.correct,
            total: result.total,
            accuracy: result.accuracy,
            by_subject: Default::default(),
        },
    );
    println!("\n{}", report.to_markdown());
    Ok(())
}
