//! Real-model gate: speculative greedy decoding equals plain greedy decoding
//! on the ternary Metal route, token for token.
//!
//! # The contract this pins
//!
//! A ternary prefill of 8-63 rows runs the tiled simdgroup GEMM
//! (`gemm_tq2_g128_simdgroup`, cosine >= 0.999 against the row-wise `v7`
//! kernel, not bit-identical), and a speculative verify is such a prefill:
//! `[next_token, draft...]` is `draft_len + 1` rows. At `draft_len` 4 the
//! verify batch is 5 rows and stays on the row-wise kernel; from `draft_len`
//! 7 it is 8 rows or more and switches kernel family, so the logits it
//! scores differ in the last bits from the single-token GEMV the plain decode
//! computes. Speculative decoding is only lossless if no argmax flips on that
//! difference, and the KV entries a tiled verify writes are read by every
//! later step. This gate decodes the same prompts both ways on the real
//! Ternary-Bonsai-1.7B and Ternary-Bonsai-8B and asserts the two token chains
//! are identical.
//!
//! # What runs
//!
//! Per model and per prompt (a chat-template prefix and a 104-token passage):
//!
//! * **Plain greedy** — the production non-speculative path
//!   (`BonsaiModel::forward_prefill`, then `forward_greedy_gpu` per token), for
//!   enough tokens to feed every run below, computed twice and required to be
//!   identical (so a divergence below cannot be run-to-run noise). EOS is
//!   ignored: both sides decode a fixed count.
//! * **Speculative greedy**, the loop `engine_greedy.rs` runs — one
//!   `forward_prefill_verify` of `[next_token, draft...]` at the position of
//!   `next_token`, accept the longest draft prefix equal to the verify
//!   argmaxes, append the argmax at the accept/reject boundary, advance by
//!   `accepted + 1` — for `draft_len` in {4, 7, 12} over 96 generated tokens.
//!   The drafts are the plain-greedy continuation (an oracle drafter, so every
//!   round verifies a full `draft_len + 1` rows), once as-is and once
//!   deterministically corrupted on three rounds in four (last, first and
//!   middle draft token), which exercises early rejection and the device-KV
//!   entries a rejected suffix leaves behind being overwritten.
//! * **Adaptive lookahead** — the draft length walks the whole range the
//!   runtime's adaptive controller can request (`AdaptiveLookaheadConfig`'s
//!   `min` 2 to `max` 12, i.e. verify batches of 3 to 13 rows), up and back
//!   down, so one generation crosses the 8-row kernel boundary in both
//!   directions; 176 generated tokens complete the walk when every draft is
//!   accepted. The controller's own EWMA is not reproduced (this crate cannot
//!   depend on the runtime): the range is what changes the kernels.
//!
//! Every verify call must complete as one batched Metal run — the process-wide
//! fused-call counter advances by exactly one — so a Metal failure that
//! silently falls back to sequential decode cannot make the comparison pass
//! vacuously. The plain decode must not advance it at all.
//!
//! On a divergence the failure names the model, prompt, mode, token index,
//! the round and batch size that produced the token, both ids and the
//! reference's top-2 margin at that step, so a near-tie flip is distinguishable
//! from a real disagreement. Nothing here loosens to make a flip pass: a flip
//! at `draft_len >= 7` is a finding about the kernel-family switch.
//!
//! # Files and records
//!
//! `Ternary-Bonsai-1.7B.gguf` and `Ternary-Bonsai-8B.gguf` are found through
//! the testkit resolver (`OXIBONSAI_MODELS_DIR`, else the workspace `models/`).
//! One test per model, so a host with only one of them cannot satisfy the
//! other's requirement: an absent file self-skips with an `executed: false`
//! `legacy-models` record, and `executed: true` (timed) is written only after
//! every assertion passed. Run in `--release`, one real-model process at a
//! time (`scripts/release-gate.sh` stage 1b does).

#![cfg(all(feature = "metal", target_os = "macos"))]

use std::collections::BTreeSet;
use std::time::Instant;

use oxibonsai_core::gguf::reader::{mmap_gguf_file, GgufFile};
use oxibonsai_kernels::dispatch::{KernelDispatcher, KernelTier};
use oxibonsai_kernels::gpu_backend::PrefillRoute;
use oxibonsai_kernels::MetalGraph;
use oxibonsai_model::model::BonsaiModel;
use oxibonsai_testkit::capability::{record_executed_timed, record_skipped, Capability};
use oxibonsai_testkit::parity::{argmax_first, gpu_serial, top_k, top_two};
use oxibonsai_testkit::workspace::{find_model, models_dir};

const TEST_TERNARY_1_7B: &str = "oxibonsai-model::speculative_ternary_metal_gates::\
                                 ternary_1_7b_speculative_greedy_equals_plain_greedy_on_metal";
const TEST_TERNARY_8B: &str = "oxibonsai-model::speculative_ternary_metal_gates::\
                               ternary_8b_speculative_greedy_equals_plain_greedy_on_metal";

/// Draft lengths of the fixed-lookahead runs: 5 rows (row-wise kernel), then
/// 8 and 13 rows (tiled kernel).
const FIXED_DRAFT_LENS: [usize; 3] = [4, 7, 12];

/// The smallest verify batch the tiled ternary GEMM serves: a mirror of
/// `PREFILL_TQ2_TILED_MIN_BATCH` (8, `gpu_backend/metal_prefill/functions.rs`
/// in the kernels crate), which is private to that crate. If the kernel
/// threshold moves, this constant and the draft lengths in
/// [`FIXED_DRAFT_LENS`] / [`ADAPTIVE_WALK`] that straddle it move with it.
const TILED_MIN_ROWS: usize = 8;

/// Tokens each fixed-lookahead run generates.
const FIXED_TOKENS: usize = 96;

/// Tokens the adaptive-lookahead run generates: enough for one full walk of
/// [`ADAPTIVE_WALK`] when every draft is accepted (160 tokens + the first).
const ADAPTIVE_TOKENS: usize = 176;

/// The largest draft length any run uses.
const MAX_DRAFT_LEN: usize = 12;

/// The adaptive controller's range (`AdaptiveLookaheadConfig::default()`:
/// `min` 2, `max` 12) walked up and back down, one draft length per round.
const ADAPTIVE_WALK: [usize; 20] = [
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3,
];

/// Context the models are loaded with: the longest prompt, the longest
/// generation and a full verify batch, with headroom.
const MAX_SEQ: usize = 512;

/// `<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n` (Qwen3
/// tokenizer): the chat-template prefix of a one-word user turn.
const PROMPT_CHAT: &[u32] = &[151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198];

/// `"Rivers carve valleys over millions of years. …"` (104 tokens, Qwen3
/// tokenizer): a prompt long enough that every verify batch attends over
/// positions beyond the first KV tile.
const PROMPT_RIVERS: &[u32] = &[
    49, 1945, 79637, 85397, 916, 11728, 315, 1635, 13, 9959, 51207, 304, 279, 1550, 8166, 11,
    85681, 4628, 438, 432, 6560, 1412, 11, 323, 23377, 9278, 323, 9798, 429, 39236, 279, 14796,
    2721, 19117, 448, 1449, 17726, 13, 10967, 279, 4268, 51039, 724, 11, 279, 1482, 69170, 323,
    21025, 1181, 2795, 11, 4752, 69125, 77366, 323, 90587, 429, 614, 22313, 9720, 2474, 279, 1156,
    23429, 82, 13, 48696, 1431, 1936, 82525, 323, 55489, 288, 311, 81823, 429, 4802, 11, 11133,
    279, 2310, 10775, 315, 17726, 323, 42801, 369, 24020, 3015, 323, 17728, 11, 323, 279, 35517,
    4226, 553, 7218, 862, 58032, 14696, 770, 13,
];

/// The two prompts every run uses, by name.
const PROMPTS: [(&str, &[u32]); 2] = [("chat", PROMPT_CHAT), ("rivers", PROMPT_RIVERS)];

/// How a speculative run chooses its draft length per round.
#[derive(Clone, Copy)]
enum Lookahead {
    /// The same draft length every round.
    Fixed(usize),
    /// [`ADAPTIVE_WALK`], one entry per round, cycled.
    Adaptive,
}

impl Lookahead {
    fn draft_len(self, round: usize) -> usize {
        match self {
            Self::Fixed(len) => len,
            Self::Adaptive => ADAPTIVE_WALK[round % ADAPTIVE_WALK.len()],
        }
    }

    fn label(self) -> String {
        match self {
            Self::Fixed(len) => format!("draft_len={len}"),
            Self::Adaptive => "adaptive 2..=12".to_string(),
        }
    }
}

/// Whether the drafts are the plain-greedy continuation as is, or corrupted.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Drafter {
    /// Every draft token is the plain-greedy token: the verify must accept
    /// them all.
    Oracle,
    /// One draft token is replaced by a different id on three rounds in four
    /// (last, first, middle): the verify must reject at exactly that token.
    Perturbed,
}

impl Drafter {
    fn label(self) -> &'static str {
        match self {
            Self::Oracle => "oracle drafts",
            Self::Perturbed => "perturbed drafts",
        }
    }

    /// The draft index replaced on `round`, if any.
    fn corrupted_index(self, round: usize, draft_len: usize) -> Option<usize> {
        if self == Self::Oracle || draft_len == 0 {
            return None;
        }
        match round % 4 {
            1 => Some(draft_len - 1),
            2 => Some(0),
            3 => Some(draft_len / 2),
            _ => None,
        }
    }
}

/// One speculative run's outcome.
struct SpecRun {
    /// Every token produced, the prefill's argmax first.
    tokens: Vec<u32>,
    /// The speculative round that produced each token (0: the prefill).
    token_round: Vec<usize>,
    /// Verify batch size, in rows, of every round.
    rows: Vec<usize>,
    /// Draft tokens proposed and accepted, summed over the rounds.
    drafted: usize,
    accepted: usize,
    /// Rounds that accepted none of their draft.
    full_rejections: usize,
    /// Rounds that accepted a strict, non-empty part of their draft.
    partial_rejections: usize,
}

/// One dense Metal fused-prefill run of `prompt`, asserted to be exactly one
/// fused batch.
fn fused_prefill(
    model: &mut BonsaiModel<'_>,
    gpu: &KernelDispatcher,
    prompt: &[u32],
    context: &str,
) -> Vec<f32> {
    let before = MetalGraph::prefill_fused_call_count();
    let logits = model
        .forward_prefill(prompt, 0, gpu)
        .unwrap_or_else(|e| panic!("{context}: prefill: {e}"));
    assert_eq!(
        MetalGraph::prefill_fused_call_count() - before,
        1,
        "{context}: the {}-token prompt did not prefill as one fused Metal batch",
        prompt.len()
    );
    logits
}

/// The production non-speculative greedy path: a fused prefill, then one
/// `forward_greedy_gpu` (GPU argmax) per token. `count` tokens, EOS ignored.
fn plain_greedy(
    model: &mut BonsaiModel<'_>,
    gpu: &KernelDispatcher,
    prompt: &[u32],
    count: usize,
    context: &str,
) -> Vec<u32> {
    model.reset();
    let logits = fused_prefill(model, gpu, prompt, context);
    let mut tokens = vec![argmax_first(&logits)];
    let decode_start = MetalGraph::prefill_fused_call_count();
    while tokens.len() < count {
        let pos = prompt.len() + tokens.len() - 1;
        let last = tokens[tokens.len() - 1];
        let next = model
            .forward_greedy_gpu(last, pos)
            .unwrap_or_else(|e| panic!("{context}: greedy decode at position {pos}: {e}"));
        tokens.push(next);
    }
    assert_eq!(
        MetalGraph::prefill_fused_call_count(),
        decode_start,
        "{context}: the plain greedy decode ran a batched prefill: it must be the single-token \
         kernel family"
    );
    tokens
}

/// The speculative loop of `engine_greedy.rs` over the real model, drafting
/// from `reference` (the plain-greedy chain) per `drafter`, until `count`
/// tokens exist.
#[allow(clippy::too_many_arguments)]
fn speculative_greedy(
    model: &mut BonsaiModel<'_>,
    gpu: &KernelDispatcher,
    prompt: &[u32],
    reference: &[u32],
    count: usize,
    lookahead: Lookahead,
    drafter: Drafter,
    vocab: u32,
    context: &str,
) -> SpecRun {
    model.reset();
    let logits = fused_prefill(model, gpu, prompt, context);
    let mut run = SpecRun {
        tokens: vec![argmax_first(&logits)],
        token_round: vec![0],
        rows: Vec::new(),
        drafted: 0,
        accepted: 0,
        full_rejections: 0,
        partial_rejections: 0,
    };
    let mut round = 0usize;
    while run.tokens.len() < count {
        round += 1;
        let draft_len = lookahead.draft_len(round - 1);
        // The reference index of the first token this round drafts.
        let first = run.tokens.len();
        let end = (first + draft_len).min(reference.len());
        let mut draft: Vec<u32> = reference[first.min(end)..end].to_vec();
        if let Some(i) = drafter.corrupted_index(round - 1, draft.len()) {
            draft[i] = (draft[i] + 1 + round as u32) % vocab;
        }
        let next_token = run.tokens[run.tokens.len() - 1];
        // `next_token` sits at this position; its KV entry is written by the
        // verify itself.
        let pos_start = prompt.len() + run.tokens.len() - 1;
        let mut batch = Vec::with_capacity(1 + draft.len());
        batch.push(next_token);
        batch.extend_from_slice(&draft);

        let before = MetalGraph::prefill_fused_call_count();
        let predictions = model
            .forward_prefill_verify(&batch, pos_start, gpu)
            .unwrap_or_else(|e| panic!("{context}: round {round}: verify: {e}"));
        assert_eq!(
            MetalGraph::prefill_fused_call_count() - before,
            1,
            "{context}: round {round}: the {}-row verify did not complete as one batched Metal \
             run (a silent fallback to sequential decode would pass vacuously)",
            batch.len()
        );
        assert_eq!(
            predictions.len(),
            batch.len(),
            "{context}: round {round}: one argmax per verified row"
        );

        let mut accepted = 0usize;
        while accepted < draft.len() && draft[accepted] == predictions[accepted] {
            accepted += 1;
        }
        run.drafted += draft.len();
        run.accepted += accepted;
        if accepted == 0 && !draft.is_empty() {
            run.full_rejections += 1;
        } else if accepted < draft.len() {
            run.partial_rejections += 1;
        }
        run.rows.push(batch.len());
        for &token in &draft[..accepted] {
            run.tokens.push(token);
            run.token_round.push(round);
        }
        // The argmax at the accept/reject boundary is the model's own next
        // token; it always exists (`predictions` has `draft.len() + 1` rows).
        run.tokens.push(predictions[accepted]);
        run.token_round.push(round);
    }
    run
}

/// Fail with a full account when `run` is not a prefix-equal copy of the
/// plain-greedy `reference`.
#[allow(clippy::too_many_arguments)]
fn assert_matches_reference(
    model: &mut BonsaiModel<'_>,
    gpu: &KernelDispatcher,
    prompt: &[u32],
    reference: &[u32],
    run: &SpecRun,
    count: usize,
    context: &str,
) {
    assert!(
        run.tokens.len() >= count && reference.len() >= run.tokens.len(),
        "{context}: produced {} tokens (wanted >= {count}), reference has {}",
        run.tokens.len(),
        reference.len()
    );
    let Some(at) = (0..run.tokens.len()).find(|&i| run.tokens[i] != reference[i]) else {
        return;
    };
    let round = run.token_round[at];
    let batch_rows = round
        .checked_sub(1)
        .and_then(|r| run.rows.get(r))
        .map_or_else(
            || "the prefill".to_string(),
            |rows| format!("a {rows}-row verify"),
        );
    // A third opinion, for scale only: the logits that predict the reference
    // token at this step, from a fused prefill of the reference context.
    model.reset();
    let replay_context: Vec<u32> = prompt
        .iter()
        .chain(reference[..at].iter())
        .copied()
        .collect();
    let margin = model
        .forward_prefill(&replay_context, 0, gpu)
        .map(|logits| {
            let two = top_two(&logits);
            format!(
                "replay top-2 gap {:e}, top-5 {}",
                two.gap(),
                top_k(&logits, 5)
            )
        })
        .unwrap_or_else(|e| format!("replay failed: {e}"));
    panic!(
        "{context}: speculative greedy diverged from plain greedy at generated token {at} \
         (round {round}, produced by {batch_rows}): speculative id {}, plain id {}; {margin}",
        run.tokens[at], reference[at]
    );
}

/// `rows` as `size x count` pairs, ascending.
fn rows_histogram(rows: &[usize]) -> String {
    let sizes: BTreeSet<usize> = rows.iter().copied().collect();
    sizes
        .iter()
        .map(|size| format!("{size}x{}", rows.iter().filter(|r| *r == size).count()))
        .collect::<Vec<_>>()
        .join(" ")
}

/// The whole matrix on one model.
fn run_gate(test_name: &str, file: &str) {
    let _gpu_guard = gpu_serial();
    let Some(path) = find_model(file) else {
        eprintln!(
            "{test_name}: {file} not found under {:?} -- skipping",
            models_dir()
        );
        record_skipped(Capability::LegacyModels, test_name);
        return;
    };
    let started = Instant::now();
    let mmap = mmap_gguf_file(&path).unwrap_or_else(|e| panic!("mmap {path:?}: {e}"));
    let gguf = GgufFile::parse(&mmap).unwrap_or_else(|e| panic!("parse {path:?}: {e}"));
    assert!(
        gguf.tensors
            .get("output.weight")
            .is_some_and(|head| head.tensor_type.is_ternary()),
        "{file}: this gate is the ternary Metal route; the file's LM head is not ternary"
    );
    let gpu = KernelDispatcher::auto_detect();
    assert_eq!(
        gpu.tier(),
        KernelTier::Gpu,
        "the speculative gate needs the Metal GPU tier on this host"
    );
    let mut model =
        BonsaiModel::from_gguf(&gguf, MAX_SEQ).unwrap_or_else(|e| panic!("load {path:?}: {e}"));
    // An all-ternary Metal engine skips the scirs2 weight upload: every fused
    // path binds the cached ternary weight set built here.
    model
        .get_or_create_gpu_cache()
        .unwrap_or_else(|e| panic!("{file}: fused weight cache: {e}"));
    // The prefill cost router may divert a prompt to sequential decode once it
    // has measured both routes; pin the fused route so the comparison isolates
    // the verify kernels from the prefill's.
    model.force_metal_prefill_route(Some(PrefillRoute::Fused));
    let vocab = u32::try_from(model.config().vocab_size).expect("the vocabulary fits u32");
    let reference_len = FIXED_TOKENS.max(ADAPTIVE_TOKENS) + MAX_DRAFT_LEN + 2;

    for (prompt_name, prompt) in PROMPTS {
        let context = format!("{file} prompt={prompt_name}");
        let reference = plain_greedy(&mut model, &gpu, prompt, reference_len, &context);
        let again = plain_greedy(&mut model, &gpu, prompt, reference_len, &context);
        assert_eq!(
            reference, again,
            "{context}: plain greedy is not reproducible run to run, so no comparison against it \
             can mean anything"
        );
        eprintln!(
            "spec-gate {context}: plain greedy {reference_len} tokens, reproducible; first ids \
             {:?}",
            &reference[..8.min(reference.len())]
        );

        let mut modes: Vec<(Lookahead, usize)> = FIXED_DRAFT_LENS
            .iter()
            .map(|&len| (Lookahead::Fixed(len), FIXED_TOKENS))
            .collect();
        modes.push((Lookahead::Adaptive, ADAPTIVE_TOKENS));
        for (lookahead, count) in modes {
            for drafter in [Drafter::Oracle, Drafter::Perturbed] {
                let label = format!("{context} {} {}", lookahead.label(), drafter.label());
                let run = speculative_greedy(
                    &mut model, &gpu, prompt, &reference, count, lookahead, drafter, vocab, &label,
                );
                assert_matches_reference(&mut model, &gpu, prompt, &reference, &run, count, &label);
                eprintln!(
                    "spec-gate {label}: {} tokens identical to plain greedy in {} rounds; \
                     verify rows [{}]; drafted {} accepted {} ({} full / {} partial rejections)",
                    run.tokens.len(),
                    run.rows.len(),
                    rows_histogram(&run.rows),
                    run.drafted,
                    run.accepted,
                    run.full_rejections,
                    run.partial_rejections,
                );
                assert_coverage(&label, lookahead, drafter, &run);
            }
        }
    }
    model.force_metal_prefill_route(None);
    record_executed_timed(Capability::LegacyModels, test_name, started.elapsed());
}

/// The run exercised the kernels and the accept/reject logic it claims to.
fn assert_coverage(label: &str, lookahead: Lookahead, drafter: Drafter, run: &SpecRun) {
    match drafter {
        Drafter::Oracle => assert_eq!(
            run.accepted, run.drafted,
            "{label}: an oracle draft is the plain-greedy chain, so every draft token must be \
             accepted"
        ),
        Drafter::Perturbed => {
            assert!(
                run.partial_rejections > 0 && run.full_rejections > 0,
                "{label}: the perturbed drafts must exercise both a partial and a full \
                 rejection ({} partial, {} full)",
                run.partial_rejections,
                run.full_rejections
            );
            assert!(run.accepted < run.drafted, "{label}: some drafts rejected");
        }
    }
    match lookahead {
        Lookahead::Fixed(len) => {
            // Every round but a truncated final one verifies a full batch.
            let full = run.rows.iter().filter(|&&rows| rows == len + 1).count();
            assert!(
                full + 1 >= run.rows.len() && full > 0,
                "{label}: verify batches were not {} rows: [{}]",
                len + 1,
                rows_histogram(&run.rows)
            );
            // `TILED_MIN_ROWS` is `PREFILL_TQ2_TILED_MIN_BATCH` (= 8, kernels
            // `gpu_backend/metal_prefill/functions.rs`): a verify of that many
            // rows or more runs the tiled GEMM family, a smaller one the
            // row-wise kernel the single-token decode shares.
            assert_eq!(
                len + 1 >= TILED_MIN_ROWS,
                run.rows.iter().any(|&rows| rows >= TILED_MIN_ROWS),
                "{label}: the tiled kernel family is expected exactly when the batch has \
                 {TILED_MIN_ROWS}+ rows"
            );
        }
        Lookahead::Adaptive if drafter == Drafter::Oracle => {
            let sizes: BTreeSet<usize> = run.rows.iter().copied().collect();
            let wanted: BTreeSet<usize> = (3..=MAX_DRAFT_LEN + 1).collect();
            assert!(
                wanted.is_subset(&sizes),
                "{label}: the adaptive walk must verify every batch size of 3..=13 rows, saw [{}]",
                rows_histogram(&run.rows)
            );
            // The walk crosses the 8-row boundary in both directions.
            let crosses = |up: bool| {
                run.rows.windows(2).any(|w| {
                    if up {
                        w[0] < TILED_MIN_ROWS && w[1] >= TILED_MIN_ROWS
                    } else {
                        w[0] >= TILED_MIN_ROWS && w[1] < TILED_MIN_ROWS
                    }
                })
            };
            assert!(
                crosses(true) && crosses(false),
                "{label}: consecutive verifies must cross the {TILED_MIN_ROWS}-row kernel \
                 boundary upward and downward: [{}]",
                run.rows
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            );
        }
        Lookahead::Adaptive => {}
    }
}

#[test]
fn ternary_1_7b_speculative_greedy_equals_plain_greedy_on_metal() {
    run_gate(TEST_TERNARY_1_7B, "Ternary-Bonsai-1.7B.gguf");
}

#[test]
fn ternary_8b_speculative_greedy_equals_plain_greedy_on_metal() {
    run_gate(TEST_TERNARY_8B, "Ternary-Bonsai-8B.gguf");
}
