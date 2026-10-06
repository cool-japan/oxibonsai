//! The M-08 long-context YaRN gate: scaled versus unscaled logits of the real
//! `Bonsai-8B.gguf` after a 20000-token natural prompt, each arm decoded in
//! its own child process (see the section comment below for the design).

use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_kernels::dispatch::{KernelDispatcher, KernelTier};
use oxibonsai_model::model::BonsaiModel;
use oxibonsai_runtime::tokenizer_bridge::TokenizerBridge;
use oxibonsai_testkit::capability::{record_executed_timed, record_skipped, Capability};
use oxibonsai_testkit::parity::{argmax_first, gpu_serial, per_step_bound, top_two};
use oxibonsai_testkit::workspace::{find_model, models_dir};

use super::TierEnvGuard;

// ═════════════════════════════════════════════════════════════════════════
//  M-08: YaRN on the real Bonsai-8B.gguf at a >= 20000-token natural prompt
// ═════════════════════════════════════════════════════════════════════════
//
// `models/Bonsai-8B.gguf` declares `qwen3.rope.scaling.{type=yarn,
// factor=4.0, original_context_length=16384}`, and `BonsaiModel::from_gguf`
// builds its RoPE table from those keys (M-08). This gate proves, on the
// real file, that the scaling reaches the model's output at a prompt past
// the original context: the SCALED arm loads the file as shipped, the
// UNSCALED arm loads a private in-memory copy whose one
// `qwen3.rope.scaling.factor` value is patched from 4.0 to 1.0
// (`oxibonsai_testkit::gguf_fixture::patch_gguf_f32_metadata`, which checks
// the key, its type and its current value before writing). Factor 1.0 is
// accepted by `RopeTable::new_with_scaling`'s YaRN branch and reproduces the
// plain table (`rope.rs`'s `new_with_scaling_yarn_factor_one_matches_plain_new`),
// so both arms go through the same `from_gguf` -> `build_rope_table` path
// and differ only in that value; the usable context comes from the
// independent `qwen3.context_length` key, identical in both.
//
// How it runs. The prompt is 20000 tokens of NATURAL English: eleven
// public-domain passages (`M08_PASSAGES`), repeated in a different
// deterministic order each round (`m08_passage_round`) and tokenized once
// by the real `tokenizer.json`. Each arm runs in its own fresh child process
// (the Metal decode path is a process-global singleton), decodes the prompt
// one token at a time through the fused Metal path (the fused batch prefill
// is slower than sequential decode at every measured prompt size on this
// hardware — see the M-18 gate in `oxibonsai-model`'s
// `legacy_parity_tests.rs`), then greedily continues `M08_CONTINUE_TOKENS`
// steps. Each child writes the full logit vector at every continuation step
// and at `M08_SAMPLED_POSITIONS` (past the 16384-token original context) to
// a dump file; the parent compares the two arms' dumps.
//
// What it asserts. At every continuation step the two arms' logits must
// differ by more than `M08_DIVERGENCE_NOISE_MULTIPLE` times the numeric
// noise band the cross-tier parity gate (`cross_tier.rs`) accepts for two
// computations of the SAME model (`per_step_bound` of the scaled step), so
// the divergence cannot be float noise. Step 0 is the clean comparison: both
// arms have consumed exactly the same 20000 tokens and differ only in the
// RoPE table. The per-step top-2 margins, the max-abs logit difference and
// whether the two continuations are still on the same history are printed
// for every row BEFORE any assertion, so a failure is diagnosable from its
// own output, and the dumps are kept on failure.
//
// Token inequality is a recorded soft check, not the assertion: both arms
// can pick the same greedy tokens while their logits differ well beyond
// noise — with a prompt of random token ids both arms fall into the same
// digit-repetition loop, which is why the prompt is natural text — so
// "the continuations differ" is reported with its margins, never required.
//
// Scope of the comparison: this is scaled-vs-unscaled inside OxiBonsai, not
// a comparison against a captured llama.cpp RoPE golden. Any later
// comparison of a real long-context run against the PrismML fork must use
// the fork's own f32 ITERATIVE RoPE cache (`theta *= theta_scale`) as the
// reference, not an f64 oracle: f32 angle quantisation alone moves a RoPE
// value by ~1.8e-3 absolute at position 30000 (ulp ~2e-3 at theta ~18139
// rad) — expected, not a bug. That gap cannot arise between this gate's two
// arms, which share one table builder.
//
// Opt-in: one run costs roughly an hour per arm on an M3 (the per-token rate
// grows with position as attention scans the whole history), so the test
// self-skips (recorded `executed: false`) unless `OXIBONSAI_M08_RUN_LONG=1`.
// `scripts/release-gate.sh` requires it by name whenever that variable is
// set, and its usage text makes that run a mandatory, separate release step.

/// Opt-in switch for [`m08_yarn_scaling_moves_real_bonsai_8b_logits_at_20000_tokens`].
const M08_RUN_LONG_ENV: &str = "OXIBONSAI_M08_RUN_LONG";
/// Child-process switches, set only by the parent test.
const M08_CHILD_ARM_ENV: &str = "OXIBONSAI_M08_CHILD_ARM";
const M08_CHILD_PROMPT_ENV: &str = "OXIBONSAI_M08_CHILD_PROMPT_IDS";
const M08_CHILD_DUMP_ENV: &str = "OXIBONSAI_M08_CHILD_DUMP";
/// This test's own function name; [`m08_libtest_name`] qualifies it with its
/// module for the child re-exec.
const M08_TEST_FN: &str = "m08_yarn_scaling_moves_real_bonsai_8b_logits_at_20000_tokens";

/// The name libtest lists this test under: its path from the test crate's
/// root. The parent re-executes this binary with `--exact <name>`, and
/// `--exact` matches the whole path, so the module the test lives in is part
/// of the name — a bare function name would select nothing, and the child
/// would exit 0 without decoding anything.
fn m08_libtest_name() -> String {
    match module_path!().split_once("::") {
        Some((_crate_name, in_crate)) => format!("{in_crate}::{M08_TEST_FN}"),
        None => M08_TEST_FN.to_string(),
    }
}

/// The re-exec name really selects this gate: listing the binary's tests
/// under `--exact <name>` yields exactly that one test. Needs no model file.
#[test]
fn m08_child_reexec_name_selects_the_gate_itself() {
    let exe = std::env::current_exe()
        .unwrap_or_else(|e| panic!("locating this test binary for the listing: {e}"));
    let name = m08_libtest_name();
    let output = std::process::Command::new(exe)
        .args([name.as_str(), "--exact", "--list"])
        .output()
        .unwrap_or_else(|e| panic!("listing this binary's tests: {e}"));
    assert!(
        output.status.success(),
        "listing failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let listing = String::from_utf8_lossy(&output.stdout);
    let listed: Vec<&str> = listing
        .lines()
        .filter_map(|line| line.strip_suffix(": test"))
        .collect();
    assert_eq!(
        listed,
        [name.as_str()],
        "`--exact {name}` must select exactly the M-08 gate; the binary lists: {listing}"
    );
}

/// The parent's re-exec reaches the child branch of the gate: a child started
/// under the qualified name with an arm set (but none of the files the real
/// child needs) fails on the first missing input, which only the gate's own
/// child branch reports. A name that selected no test would instead exit 0
/// having run nothing. Needs no model file.
#[test]
fn m08_child_reexec_enters_the_child_branch_of_the_gate() {
    let run = oxibonsai_testkit::parity::run_named_test_in_child(
        &m08_libtest_name(),
        &[(M08_CHILD_ARM_ENV, "reexec-probe")],
        std::time::Duration::from_secs(300),
    )
    .unwrap_or_else(|e| panic!("spawning the M-08 probe child: {e}"));
    assert!(
        run.status.is_some(),
        "the probe child did not exit in time: {}",
        run.output
    );
    assert!(
        !run.succeeded(),
        "the probe child must fail on its missing prompt file, not pass by running no test: {}",
        run.output
    );
    let expected = format!("{M08_CHILD_PROMPT_ENV} is not set in the M-08 reexec-probe child");
    assert!(
        run.output.contains(&expected),
        "the probe child did not reach the gate's child branch (expected {expected:?}): {}",
        run.output
    );
}

/// Prompt length: past `qwen3.rope.scaling.original_context_length` (16384).
const M08_PROMPT_TOKENS: usize = 20_000;
/// Greedy continuation steps compared after the prompt.
const M08_CONTINUE_TOKENS: usize = 16;
/// Context budget: the prompt plus the continuation, with headroom.
const M08_MAX_SEQ: usize = M08_PROMPT_TOKENS + 64;
/// Prompt positions whose logits are also compared (recorded, not
/// asserted): the last position inside the original 16384-token context and
/// four past it. Each must be below `M08_PROMPT_TOKENS - 1`; the logits at
/// the final prompt position are continuation step 0.
const M08_SAMPLED_POSITIONS: [usize; 5] = [16_383, 16_384, 17_408, 18_432, 19_456];
/// Progress-line cadence inside a child.
const M08_PROGRESS_EVERY: usize = 1_000;
/// Hang guard per child — roughly three times the measured per-arm time.
const M08_CHILD_DEADLINE: std::time::Duration = std::time::Duration::from_secs(10_800);
/// The YaRN factor key the unscaled arm patches, and its shipped value.
const M08_FACTOR_KEY: &str = "qwen3.rope.scaling.factor";
const M08_SHIPPED_FACTOR: f32 = 4.0;

/// The stated divergence threshold: at every continuation step the two
/// arms' max-abs logit difference must exceed this many times
/// [`per_step_bound`] of the scaled step — the largest disagreement the
/// cross-backend parity gate accepts as float noise between two
/// computations of the same model (`5e-3 * max(1, max|logit|)`, itself 4x
/// the worst cross-backend deviation measured on the real models, see
/// `REL_TOL`). 5x that bound is ~20x the worst measured float noise.
///
/// Calibrated on this file and this prompt: at a 3072-token prefix of the
/// same natural prompt, the 16 continuation steps measured 15.2x-43.0x the
/// bound (max-abs logit difference 1.61-5.18 against bounds of 0.094-0.157,
/// identical greedy tokens in both arms), so 5x keeps 3x headroom below the
/// smallest divergence observed.
const M08_DIVERGENCE_NOISE_MULTIPLE: f32 = 5.0;

/// Dump-file layout: this magic, a `u32` vocabulary size, then records of
/// `kind: u8`, `index: u32` (a prompt position or a continuation step) and
/// `vocab` `f32` logits, all little-endian.
const M08_DUMP_MAGIC: &[u8; 8] = b"OXM08LG1";
const M08_KIND_PROMPT: u8 = b'P';
const M08_KIND_CONTINUATION: u8 = b'C';

/// Eleven public-domain passages (a prime count, so every stride in
/// `1..11` yields a full permutation in [`m08_passage_round`]): the US
/// Declaration of Independence (1776), the Gettysburg Address (1863), the
/// Preamble to the US Constitution (1787), and the openings of A Tale of Two
/// Cities (1859), Pride and Prejudice (1813), Moby-Dick (1851), Walden
/// (1854), Lincoln's Second Inaugural Address (1865) and Alice's Adventures
/// in Wonderland (1865).
const M08_PASSAGES: [&str; 11] = [
    "When in the Course of human events, it becomes necessary for one people to dissolve the political bands which have connected them with another, and to assume among the powers of the earth, the separate and equal station to which the Laws of Nature and of Nature's God entitle them, a decent respect to the opinions of mankind requires that they should declare the causes which impel them to the separation.",
    "We hold these truths to be self-evident, that all men are created equal, that they are endowed by their Creator with certain unalienable Rights, that among these are Life, Liberty and the pursuit of Happiness. That to secure these rights, Governments are instituted among Men, deriving their just powers from the consent of the governed.",
    "Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battle-field of that war.",
    "But, in a larger sense, we can not dedicate—we can not consecrate—we can not hallow—this ground. The brave men, living and dead, who struggled here, have consecrated it, far above our poor power to add or detract. The world will little note, nor long remember what we say here, but it can never forget what they did here.",
    "We the People of the United States, in Order to form a more perfect Union, establish Justice, insure domestic Tranquility, provide for the common defence, promote the general Welfare, and secure the Blessings of Liberty to ourselves and our Posterity, do ordain and establish this Constitution for the United States of America.",
    "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair, we had everything before us, we had nothing before us.",
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife. However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered the rightful property of some one or other of their daughters.",
    "Call me Ishmael. Some years ago—never mind how long precisely—having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world. It is a way I have of driving off the spleen and regulating the circulation.",
    "I went to the woods because I wished to live deliberately, to front only the essential facts of life, and see if I could not learn what it had to teach, and not, when I came to die, discover that I had not lived. I did not wish to live what was not life, living is so dear; nor did I wish to practise resignation, unless it was quite necessary.",
    "With malice toward none, with charity for all, with firmness in the right as God gives us to see the right, let us strive on to finish the work we are in, to bind up the nation's wounds, to care for him who shall have borne the battle and for his widow and his orphan, to do all which may achieve and cherish a just and lasting peace among ourselves and with all nations.",
    "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, 'and what is the use of a book,' thought Alice 'without pictures or conversations?'",
];

/// Round `round` of the prompt: all eleven passages, each exactly once, in
/// the order `i -> (stride * i + offset) mod 11` with a stride and offset
/// that change every round, joined by blank lines. Deterministic.
fn m08_passage_round(round: usize) -> String {
    let n = M08_PASSAGES.len();
    let stride = 1 + round % (n - 1);
    let offset = (3 * round) % n;
    (0..n)
        .map(|i| M08_PASSAGES[(stride * i + offset) % n])
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// The gate's prompt: enough rounds of [`m08_passage_round`] to pass
/// [`M08_PROMPT_TOKENS`] once tokenized as ONE text, cut to exactly that
/// many tokens.
fn m08_natural_prompt_ids(tokenizer: &TokenizerBridge) -> Vec<u32> {
    let per_round = tokenizer
        .encode(&m08_passage_round(0))
        .expect("encode one round of the M-08 passages")
        .len();
    assert!(per_round > 0, "one round of passages encoded to no tokens");
    let rounds = M08_PROMPT_TOKENS / per_round + 2;
    let text = (0..rounds)
        .map(m08_passage_round)
        .collect::<Vec<_>>()
        .join("\n\n");
    let mut ids = tokenizer.encode(&text).expect("encode the M-08 prompt");
    assert!(
        ids.len() >= M08_PROMPT_TOKENS,
        "{rounds} rounds of passages encoded to only {} tokens, fewer than {M08_PROMPT_TOKENS}",
        ids.len()
    );
    ids.truncate(M08_PROMPT_TOKENS);
    ids
}

/// Every round of the M-08 prompt holds each passage exactly once, and the
/// order really does change from round to round — the property the prompt
/// relies on to be natural text that is not one short period repeated. No
/// model file needed.
#[test]
fn m08_prompt_rounds_are_permutations_of_every_passage() {
    let first_rounds: Vec<String> = (0..10).map(m08_passage_round).collect();
    for (round, text) in first_rounds.iter().enumerate() {
        let paragraphs: Vec<&str> = text.split("\n\n").collect();
        assert_eq!(paragraphs.len(), M08_PASSAGES.len(), "round {round}");
        for passage in &M08_PASSAGES {
            assert_eq!(
                paragraphs.iter().filter(|p| *p == passage).count(),
                1,
                "round {round} must hold every passage exactly once"
            );
        }
    }
    let distinct: std::collections::HashSet<&String> = first_rounds.iter().collect();
    assert_eq!(
        distinct.len(),
        first_rounds.len(),
        "the first ten rounds must all use different passage orders"
    );
    for pos in M08_SAMPLED_POSITIONS {
        assert!(
            pos + 1 < M08_PROMPT_TOKENS,
            "sampled position {pos} must precede the final prompt position"
        );
    }
}

/// Writes one child's logit records (see [`M08_DUMP_MAGIC`] for the layout).
struct M08DumpWriter {
    file: std::io::BufWriter<std::fs::File>,
    vocab: Option<usize>,
}

impl M08DumpWriter {
    fn create(path: &std::path::Path) -> std::io::Result<Self> {
        Ok(Self {
            file: std::io::BufWriter::new(std::fs::File::create(path)?),
            vocab: None,
        })
    }

    fn record(&mut self, kind: u8, index: usize, logits: &[f32]) -> std::io::Result<()> {
        use std::io::Write;
        match self.vocab {
            None => {
                self.file.write_all(M08_DUMP_MAGIC)?;
                self.file.write_all(&u32_le(logits.len()))?;
                self.vocab = Some(logits.len());
            }
            Some(vocab) if vocab != logits.len() => {
                return Err(std::io::Error::other(format!(
                    "logit vector length changed from {vocab} to {}",
                    logits.len()
                )));
            }
            Some(_) => {}
        }
        self.file.write_all(&[kind])?;
        self.file.write_all(&u32_le(index))?;
        for &value in logits {
            self.file.write_all(&value.to_le_bytes())?;
        }
        Ok(())
    }

    fn finish(mut self) -> std::io::Result<()> {
        use std::io::Write;
        self.file.flush()
    }
}

/// `value` as 4 little-endian bytes; every count this file writes is far
/// below `u32::MAX`, which the conversion checks.
fn u32_le(value: usize) -> [u8; 4] {
    u32::try_from(value)
        .unwrap_or_else(|_| panic!("{value} does not fit in a u32"))
        .to_le_bytes()
}

/// One logit record read back from a child's dump.
struct M08Record {
    kind: u8,
    index: usize,
    logits: Vec<f32>,
}

fn m08_read_dump(path: &std::path::Path) -> Vec<M08Record> {
    let bytes =
        std::fs::read(path).unwrap_or_else(|e| panic!("read M-08 dump {}: {e}", path.display()));
    let read_u32 = |at: usize| -> usize {
        let chunk: [u8; 4] = bytes
            .get(at..at + 4)
            .and_then(|s| s.try_into().ok())
            .unwrap_or_else(|| panic!("M-08 dump {} truncated at byte {at}", path.display()));
        u32::from_le_bytes(chunk) as usize
    };
    assert!(
        bytes.starts_with(M08_DUMP_MAGIC),
        "M-08 dump {} does not start with the expected magic",
        path.display()
    );
    let vocab = read_u32(M08_DUMP_MAGIC.len());
    let mut at = M08_DUMP_MAGIC.len() + 4;
    let mut records = Vec::new();
    while at < bytes.len() {
        let kind = bytes[at];
        let index = read_u32(at + 1);
        let start = at + 5;
        let end = start + 4 * vocab;
        let payload = bytes.get(start..end).unwrap_or_else(|| {
            panic!(
                "M-08 dump {} truncated inside record {}",
                path.display(),
                records.len()
            )
        });
        let logits = payload
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        records.push(M08Record {
            kind,
            index,
            logits,
        });
        at = end;
    }
    records
}

fn m08_read_ids(path: &std::path::Path) -> Vec<u32> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|e| panic!("read M-08 prompt ids {}: {e}", path.display()));
    assert_eq!(bytes.len() % 4, 0, "M-08 prompt ids file is not whole u32s");
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Loads `arm`'s model — the shipped file, or the copy with the YaRN factor
/// patched to 1.0 — on the auto-detected GPU tier with its weights
/// uploaded, as `InferenceEngine::from_gguf` does for this one-bit file.
/// The bytes and the parsed file are leaked: this runs only inside a
/// short-lived child process dedicated to one arm.
fn m08_load_arm(arm: &str) -> (BonsaiModel<'static>, KernelDispatcher) {
    let model_path = find_model("Bonsai-8B.gguf")
        .unwrap_or_else(|| panic!("M-08 {arm} child could not find Bonsai-8B.gguf"));
    let mut bytes = std::fs::read(&model_path).expect("read real Bonsai-8B.gguf");
    match arm {
        "scaled" => {}
        "unscaled" => oxibonsai_testkit::gguf_fixture::patch_gguf_f32_metadata(
            &mut bytes,
            M08_FACTOR_KEY,
            M08_SHIPPED_FACTOR,
            1.0,
        )
        .unwrap_or_else(|e| panic!("patch {M08_FACTOR_KEY} to 1.0: {e}")),
        other => panic!("unknown M-08 arm {other:?}"),
    }
    let bytes: &'static [u8] = Box::leak(bytes.into_boxed_slice());
    let gguf: &'static GgufFile<'static> = Box::leak(Box::new(
        GgufFile::parse(bytes).expect("parse Bonsai-8B.gguf"),
    ));
    let factor = gguf
        .metadata
        .get_f32(M08_FACTOR_KEY)
        .expect("the YaRN factor key is present");
    println!("M08_ARM arm={arm} {M08_FACTOR_KEY}={factor}");

    let kernel = KernelDispatcher::auto_detect();
    assert_eq!(
        kernel.tier(),
        KernelTier::Gpu,
        "KernelDispatcher::auto_detect() did not select the GPU tier in the M-08 {arm} child"
    );
    let mut model = BonsaiModel::from_gguf(gguf, M08_MAX_SEQ).expect("BonsaiModel::from_gguf");
    model.upload_weights_to_gpu(&kernel);
    let effective_context = model.max_context();
    println!("M08_EFFECTIVE_CONTEXT arm={arm} value={effective_context}");
    assert!(
        effective_context >= M08_MAX_SEQ,
        "M-08 {arm} child's effective context {effective_context} is below {M08_MAX_SEQ}"
    );
    (model, kernel)
}

/// One arm's work, inside its own child process: decode the prompt token by
/// token, record the sampled positions and every continuation step, print
/// the human-readable lines.
fn run_m08_child(arm: &str) {
    let prompt_path = std::env::var(M08_CHILD_PROMPT_ENV)
        .unwrap_or_else(|_| panic!("{M08_CHILD_PROMPT_ENV} is not set in the M-08 {arm} child"));
    let dump_path = std::env::var(M08_CHILD_DUMP_ENV)
        .unwrap_or_else(|_| panic!("{M08_CHILD_DUMP_ENV} is not set in the M-08 {arm} child"));
    let prompt = m08_read_ids(std::path::Path::new(&prompt_path));
    assert!(
        prompt.len() >= 2,
        "the M-08 prompt needs at least two tokens"
    );

    let (mut model, kernel) = m08_load_arm(arm);
    let mut dump = M08DumpWriter::create(std::path::Path::new(&dump_path))
        .unwrap_or_else(|e| panic!("create M-08 dump {dump_path}: {e}"));

    let start = std::time::Instant::now();
    let mut logits = Vec::new();
    for (pos, &token) in prompt.iter().enumerate() {
        logits = model
            .forward(token, pos, &kernel)
            .unwrap_or_else(|e| panic!("M-08 {arm} prompt position {pos}: {e}"));
        if pos == 0 {
            assert!(
                model.gpu_path_active(),
                "M-08 {arm}: the first token did not run on the fused Metal decode path"
            );
        }
        if pos + 1 < prompt.len() && M08_SAMPLED_POSITIONS.contains(&pos) {
            dump.record(M08_KIND_PROMPT, pos, &logits)
                .unwrap_or_else(|e| panic!("write M-08 dump: {e}"));
            let top = top_two(&logits);
            println!(
                "M08_SAMPLE arm={arm} pos={pos} argmax={} top2_margin={:.6}",
                top.index(),
                top.gap()
            );
        }
        if pos > 0 && pos % M08_PROGRESS_EVERY == 0 {
            let elapsed = start.elapsed();
            println!(
                "M08_PROGRESS arm={arm} pos={pos} elapsed_ms={} us_per_token={:.1}",
                elapsed.as_millis(),
                elapsed.as_micros() as f64 / pos as f64
            );
        }
    }
    println!(
        "M08_PROMPT_DONE arm={arm} tokens={} elapsed_ms={}",
        prompt.len(),
        start.elapsed().as_millis()
    );

    let mut generated = Vec::with_capacity(M08_CONTINUE_TOKENS);
    for step in 0..M08_CONTINUE_TOKENS {
        let top = top_two(&logits);
        let token = argmax_first(&logits);
        dump.record(M08_KIND_CONTINUATION, step, &logits)
            .unwrap_or_else(|e| panic!("write M-08 dump: {e}"));
        println!(
            "M08_STEP arm={arm} step={step} token={token} top2_margin={:.6}",
            top.gap()
        );
        generated.push(token);
        if step + 1 < M08_CONTINUE_TOKENS {
            logits = model
                .forward(token, prompt.len() + step, &kernel)
                .unwrap_or_else(|e| panic!("M-08 {arm} continuation step {step}: {e}"));
        }
    }
    assert!(
        model.gpu_path_active(),
        "M-08 {arm}: the decode finished off the fused Metal path"
    );
    dump.finish()
        .unwrap_or_else(|e| panic!("flush M-08 dump {dump_path}: {e}"));
    println!(
        "M08_TOKENS arm={arm} tokens={}",
        generated
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",")
    );
}

/// One compared row: a sampled prompt position or a continuation step.
struct M08Row {
    label: String,
    continuation: bool,
    same_history: bool,
    scaled_argmax: usize,
    scaled_margin: f32,
    unscaled_argmax: usize,
    unscaled_margin: f32,
    max_abs_diff: f32,
    noise_bound: f32,
}

impl M08Row {
    fn threshold(&self) -> f32 {
        M08_DIVERGENCE_NOISE_MULTIPLE * self.noise_bound
    }
}

/// M-08: YaRN scaling changes the real `Bonsai-8B.gguf`'s logits beyond
/// float noise at every greedy continuation step after a 20000-token natural
/// prompt (see this section's header comment for the design, the stated
/// threshold and why token inequality is only a recorded soft check).
/// Opt-in via `OXIBONSAI_M08_RUN_LONG=1`; otherwise self-skips with an
/// `executed: false` record.
#[test]
fn m08_yarn_scaling_moves_real_bonsai_8b_logits_at_20000_tokens() {
    const TEST: &str = "oxibonsai-runtime::legacy_parity_tests::\
                        m08_yarn_scaling_moves_real_bonsai_8b_logits_at_20000_tokens";
    let _tier_env = TierEnvGuard::cleared();

    if let Ok(arm) = std::env::var(M08_CHILD_ARM_ENV) {
        run_m08_child(&arm);
        return;
    }

    let _gpu = gpu_serial();
    let gate_start = std::time::Instant::now();
    if find_model("Bonsai-8B.gguf").is_none() {
        eprintln!("skip: Bonsai-8B.gguf not found under {:?}", models_dir());
        record_skipped(Capability::LegacyModels, TEST);
        return;
    }
    let Some(tokenizer_path) = find_model("tokenizer.json") else {
        eprintln!("skip: tokenizer.json not found under {:?}", models_dir());
        record_skipped(Capability::LegacyModels, TEST);
        return;
    };
    if std::env::var(M08_RUN_LONG_ENV).as_deref() != Ok("1") {
        eprintln!(
            "skip: {M08_RUN_LONG_ENV}=1 is not set -- this real {M08_PROMPT_TOKENS}-token decode \
             takes about an hour per arm and is a separate release step"
        );
        record_skipped(Capability::LegacyModels, TEST);
        return;
    }

    let tokenizer = TokenizerBridge::from_file(
        tokenizer_path
            .to_str()
            .expect("models/ path is valid UTF-8"),
    )
    .expect("load real tokenizer.json");
    let prompt = m08_natural_prompt_ids(&tokenizer);
    let excerpt = |ids: &[u32]| tokenizer.decode(ids).unwrap_or_else(|e| format!("<{e}>"));
    eprintln!(
        "M08_PROMPT tokens={} head={:?} tail={:?}",
        prompt.len(),
        excerpt(&prompt[..48]),
        excerpt(&prompt[prompt.len() - 48..])
    );
    let prompt_bytes: Vec<u8> = prompt.iter().flat_map(|id| id.to_le_bytes()).collect();
    let prompt_file = oxibonsai_testkit::temp_path::TempFile::write(
        "oxibonsai-m08-prompt",
        ".ids",
        &prompt_bytes,
    )
    .expect("write the M-08 prompt ids");
    let prompt_path = prompt_file
        .path()
        .to_str()
        .expect("temp dir path is valid UTF-8")
        .to_string();

    let mut dumps = Vec::new();
    let mut tokens_by_arm = Vec::new();
    for arm in ["scaled", "unscaled"] {
        let dump_path = oxibonsai_testkit::temp_path::unique_path(
            &format!("oxibonsai-m08-{arm}-logits"),
            ".bin",
        );
        let dump_str = dump_path
            .to_str()
            .expect("temp dir path is valid UTF-8")
            .to_string();
        let run = oxibonsai_testkit::parity::run_named_test_in_child(
            &m08_libtest_name(),
            &[
                (M08_CHILD_ARM_ENV, arm),
                (M08_CHILD_PROMPT_ENV, prompt_path.as_str()),
                (M08_CHILD_DUMP_ENV, dump_str.as_str()),
            ],
            M08_CHILD_DEADLINE,
        )
        .unwrap_or_else(|e| panic!("spawn the M-08 {arm} child: {e}"));
        eprint!("{}", run.output);
        match run.status {
            None => panic!(
                "the M-08 {arm} child did not finish within {M08_CHILD_DEADLINE:?} and was killed"
            ),
            Some(status) => assert!(
                status.success(),
                "the M-08 {arm} child exited with {status:?}; its output is above"
            ),
        }
        let prefix = format!("M08_TOKENS arm={arm} tokens=");
        let tokens: Vec<u32> = run
            .output
            .lines()
            .find_map(|line| line.strip_prefix(prefix.as_str()))
            .unwrap_or_else(|| panic!("the M-08 {arm} child printed no M08_TOKENS line"))
            .split(',')
            .filter(|s| !s.is_empty())
            .map(|s| s.parse().expect("an M08_TOKENS entry is a u32"))
            .collect();
        assert_eq!(
            tokens.len(),
            M08_CONTINUE_TOKENS,
            "{arm} continuation length"
        );
        tokens_by_arm.push(tokens);
        dumps.push(dump_path);
    }

    let scaled = m08_read_dump(&dumps[0]);
    let unscaled = m08_read_dump(&dumps[1]);
    let expected_records = M08_SAMPLED_POSITIONS.len() + M08_CONTINUE_TOKENS;
    assert_eq!(scaled.len(), expected_records, "scaled dump record count");
    assert_eq!(
        unscaled.len(),
        expected_records,
        "unscaled dump record count"
    );

    let (scaled_tokens, unscaled_tokens) = (&tokens_by_arm[0], &tokens_by_arm[1]);
    let mut rows = Vec::with_capacity(expected_records);
    for (s, u) in scaled.iter().zip(&unscaled) {
        assert!(
            s.kind == u.kind && s.index == u.index && s.logits.len() == u.logits.len(),
            "the two M-08 dumps are not aligned record for record"
        );
        let continuation = s.kind == M08_KIND_CONTINUATION;
        let (label, same_history) = if continuation {
            (
                format!("step {}", s.index),
                scaled_tokens[..s.index] == unscaled_tokens[..s.index],
            )
        } else {
            (format!("pos {}", s.index), true)
        };
        let max_abs_diff = s
            .logits
            .iter()
            .zip(&u.logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let scaled_absmax = s.logits.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let (st, ut) = (top_two(&s.logits), top_two(&u.logits));
        rows.push(M08Row {
            label,
            continuation,
            same_history,
            scaled_argmax: st.index(),
            scaled_margin: st.gap(),
            unscaled_argmax: ut.index(),
            unscaled_margin: ut.gap(),
            max_abs_diff,
            noise_bound: per_step_bound(scaled_absmax),
        });
    }

    eprintln!(
        "M08_TABLE row | history | scaled argmax (top2 margin) | unscaled argmax (top2 margin) \
         | max|dlogit| | noise bound | x bound"
    );
    for row in &rows {
        eprintln!(
            "M08_ROW {} | {} | {} ({:.4}) | {} ({:.4}) | {:.4} | {:.4} | {:.1}",
            row.label,
            if row.same_history { "same" } else { "diverged" },
            row.scaled_argmax,
            row.scaled_margin,
            row.unscaled_argmax,
            row.unscaled_margin,
            row.max_abs_diff,
            row.noise_bound,
            row.max_abs_diff / row.noise_bound
        );
    }
    let decode = |ids: &[u32]| tokenizer.decode(ids).unwrap_or_else(|e| format!("<{e}>"));
    match scaled_tokens
        .iter()
        .zip(unscaled_tokens)
        .position(|(a, b)| a != b)
    {
        Some(step) => eprintln!(
            "M08_SOFT_CHECK the greedy continuations diverge at step {step}: scaled={:?} \
             unscaled={:?}",
            decode(scaled_tokens),
            decode(unscaled_tokens)
        ),
        None => eprintln!(
            "M08_SOFT_CHECK the greedy continuations are identical ({:?}); the logit rows above \
             carry the divergence and each step's top-2 margins",
            decode(scaled_tokens)
        ),
    }

    let below: Vec<&M08Row> = rows
        .iter()
        .filter(|row| row.continuation && row.max_abs_diff <= row.threshold())
        .collect();
    if !below.is_empty() {
        panic!(
            "YaRN scaling did not move the real Bonsai-8B.gguf's logits beyond \
             {M08_DIVERGENCE_NOISE_MULTIPLE}x the numeric noise band at {} continuation step(s) \
             after a {M08_PROMPT_TOKENS}-token prompt (first: {}, max|dlogit| {:.4} <= threshold \
             {:.4}). Dumps kept at {:?}; the full table is above.",
            below.len(),
            below[0].label,
            below[0].max_abs_diff,
            below[0].threshold(),
            dumps
        );
    }

    for dump in &dumps {
        // Best effort: a leftover dump in the temp dir is harmless.
        let _ = std::fs::remove_file(dump);
    }
    let elapsed = gate_start.elapsed();
    record_executed_timed(Capability::Metal, TEST, elapsed);
    record_executed_timed(Capability::LegacyModels, TEST, elapsed);
}
