//! Design SS8.2 **G6** and **G7** against the real Bonsai 2 27B GGUF's own
//! metadata: **G6** is tokenization exactness against the real vocabulary
//! and pre-tokenizer (`pre = qwen35`), for all 5 golden texts — including
//! the emoji byte-fallback split and `<think>` resolving to id 248068.
//! **G7** is chat-template rendering, byte-identical to the fork's own
//! `/apply-template` dump, for the fork's 5 rendering cases (plain,
//! `enable_thinking=false`, `reasoning_effort=low`, a multi-turn history
//! with `reasoning_content`, and a tool-call schema).
//!
//! # Why a header-only real-file test
//!
//! `GgufFile::parse` over an `mmap` reads the header, the metadata KV table
//! (which carries the full `tokenizer.ggml.tokens`/`tokenizer.ggml.merges`
//! arrays for this architecture) and the tensor descriptors — never a
//! tensor's bytes. Building the tokenizer this way costs on the order of
//! 100ms even for the 27B file and needs none of its multi-GB weight data,
//! so this file's cases are NOT `default-filter`-excluded from a plain
//! `cargo nextest run` (unlike `hybrid_forward_parity_tests`'s and
//! `bonsai2_engine_tests`'s full-decode gates): they cost about as much as
//! any other test.
//!
//! # What is asserted
//!
//! The 5 golden texts and their exact token id sequences below are the
//! `backend == "metal"` `/tokenize` dump of the PrismML llama.cpp fork
//! (`add_special=False`), covering: plain ASCII, source code, a
//! comma-heavy sentence, a mixed ASCII/Japanese/emoji/whitespace corpus
//! (whose emoji splits into three byte-fallback ids: 10838 spans a leading
//! space plus the emoji's first two UTF-8 bytes, then 235 and 96 are its
//! third and fourth bytes individually), and a `<think>`-tagged chat turn
//! (`<think>` itself is vocabulary id 248068, one of the special tokens in
//! the model's 248044..248076 range, not a byte-fallback sequence).
//!

use oxibonsai_core::gguf::reader::{mmap_gguf_file, GgufFile};
use oxibonsai_runtime::engine::tokenizer_from_gguf;
use oxibonsai_runtime::reasoning::{split_reasoning, ReasoningChunk, ReasoningSplitter};
use oxibonsai_testkit::capability::{record_executed, record_skipped, Capability};
use oxibonsai_tokenizer::chat_templates::{RenderMessage, RenderOptions, ResolvedChatTemplate};

/// `OXI_BONSAI2_PQ2_GGUF`, else `$OXIBONSAI_MODELS_DIR/Ternary-Bonsai-2-27B-PQ2_0.gguf`,
/// else `<workspace>/models/Ternary-Bonsai-2-27B-PQ2_0.gguf`.
///
/// The workspace-relative fallback is safe as a *default* (unlike the
/// full-weight-load real-27B gates elsewhere in this workspace, which
/// deliberately have no such fallback and rely only on an explicit env var
/// — see `hybrid_forward_parity_tests.rs`/`bonsai2_engine_tests.rs`) because
/// every case in this file only `mmap`s the file and parses its header and
/// metadata KV table, never a tensor's bytes: it costs on the order of
/// 100ms even for this file's multi-GB size, so a plain `cargo test`/`cargo
/// nextest run` that happens to find the real weights under `models/` does
/// not pay a full-weight-load cost by accident.
///
/// Set `OXI_REQUIRE_MODEL_FILES=1` to turn a missing file into a hard
/// failure instead of a skip — the same convention
/// `bonsai2_real/harness.rs::locate_model` (model crate) and
/// `model_registry.rs`'s real-file tests use, so a CI run that mounts the
/// weights can demand every gate actually ran instead of silently skipping.
fn locate_27b_gguf(test_name: &str) -> Option<std::path::PathBuf> {
    let require_real_files = std::env::var("OXI_REQUIRE_MODEL_FILES")
        .map(|v| v == "1")
        .unwrap_or(false);
    if let Ok(path) = std::env::var("OXI_BONSAI2_PQ2_GGUF") {
        if !path.is_empty() {
            let path = std::path::PathBuf::from(path);
            if path.is_file() {
                return Some(path);
            }
        }
    }
    if let Ok(dir) = std::env::var("OXIBONSAI_MODELS_DIR") {
        if !dir.is_empty() {
            let path = std::path::PathBuf::from(dir).join("Ternary-Bonsai-2-27B-PQ2_0.gguf");
            if path.is_file() {
                return Some(path);
            }
        }
    }
    // `CARGO_MANIFEST_DIR` is fixed at this crate's own compile time
    // (`<workspace-root>/crates/oxibonsai-runtime`); every workspace member
    // lives exactly two directories below the root, so this always resolves
    // to the workspace's own (gitignored, often-absent-in-a-worktree)
    // `models/` directory regardless of which directory the test binary was
    // invoked from — the same technique `oxibonsai_testkit::workspace::root`
    // uses.
    let workspace_models = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../models")
        .join("Ternary-Bonsai-2-27B-PQ2_0.gguf");
    if workspace_models.is_file() {
        return Some(workspace_models);
    }
    assert!(
        !require_real_files,
        "OXI_REQUIRE_MODEL_FILES=1: {test_name} needs a real Bonsai 2 27B GGUF; set \
         OXI_BONSAI2_PQ2_GGUF or OXIBONSAI_MODELS_DIR"
    );
    eprintln!(
        "skip {test_name}: no real Bonsai 2 27B GGUF located (set OXI_BONSAI2_PQ2_GGUF or \
         OXIBONSAI_MODELS_DIR)"
    );
    record_skipped(Capability::Bonsai2Models, test_name);
    None
}

/// One golden `(text, exact token id sequence)` pair — the fork's
/// `/tokenize` `add_special=False` dump, embedded verbatim (ids only; the
/// fork's own `piece` field is redundant with decoding the id through this
/// same vocabulary, so it is not duplicated here).
struct GoldenTokenization {
    text: &'static str,
    ids: &'static [u32],
}

const GOLDEN_TOKENIZATIONS: [GoldenTokenization; 5] = [
    GoldenTokenization {
        text: "The capital of Japan is",
        ids: &[760, 6511, 314, 6124, 369],
    },
    GoldenTokenization {
        text: "def fibonacci(n):",
        ids: &[727, 73111, 1393, 1590],
    },
    GoldenTokenization {
        text: "Once upon a time, in a small village by the sea,",
        ids: &[
            12162, 5028, 264, 854, 11, 303, 264, 2526, 13721, 539, 279, 9117, 11,
        ],
    },
    // Mixed ASCII/Japanese/emoji/whitespace. The 🍣 emoji (U+1F363, UTF-8
    // bytes F0 9F 8D A3) does not round-trip through the byte-level
    // vocabulary as one piece: id 10838 covers three bytes — a leading
    // space (0x20) plus the emoji's own first two bytes (0xF0 0x9F) — then
    // 235 and 96 are its third and fourth bytes (0x8D 0xA3) individually —
    // the byte-fallback split this gate exists to pin.
    GoldenTokenization {
        text: "Hello, world! 日本語のテキストです。 🍣 test123 can't won't  double  space\n\nnewlines\tTab",
        ids: &[
            9419, 11, 1814, 0, 220, 247359, 15303, 210342, 36298, 1710, 10838, 235, 96, 1228, 16,
            17, 18, 628, 914, 2677, 914, 220, 1923, 220, 3433, 271, 902, 7718, 197, 8320,
        ],
    },
    // `<think>` (id 248068) is a real vocabulary entry, not a byte-fallback
    // sequence — the design's own think-block id.
    GoldenTokenization {
        text: "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n",
        ids: &[
            248045, 846, 198, 12675, 248046, 198, 248045, 74455, 198, 248068, 198,
        ],
    },
];

/// The design's own `<think>` id, asserted directly against the golden data
/// above (design SS7.0: "special tokens 248044..248076 incl. ... `<think>`").
const THINK_TOKEN_ID: u32 = 248068;

#[test]
fn real_27b_tokenize_matches_the_fork_for_all_five_golden_texts_bonsai2() {
    const TEST: &str = "oxibonsai-runtime::bonsai2_runtime_tests::\
                        real_27b_tokenize_matches_the_fork_for_all_five_golden_texts_bonsai2";
    let Some(path) = locate_27b_gguf(TEST) else {
        return;
    };

    let mmap = mmap_gguf_file(&path).unwrap_or_else(|e| panic!("mmap {}: {e}", path.display()));
    let gguf = GgufFile::parse(&mmap).expect("real 27B GGUF header parses");
    let tokenizer = tokenizer_from_gguf(&gguf).expect("GGUF-embedded qwen35 tokenizer builds");

    let mut failures = Vec::new();
    for case in &GOLDEN_TOKENIZATIONS {
        let ids = match tokenizer.encode(case.text) {
            Ok(ids) => ids,
            Err(e) => {
                failures.push(format!("{:?}: encode failed: {e}", case.text));
                continue;
            }
        };
        if ids != case.ids {
            failures.push(format!(
                "{:?}: encode id mismatch\n  ours:  {ids:?}\n  fork:  {:?}",
                case.text, case.ids
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "real 27B tokenize (G6) diverged from the fork:\n{}",
        failures.join("\n")
    );

    let think_case = GOLDEN_TOKENIZATIONS
        .iter()
        .find(|c| c.ids.contains(&THINK_TOKEN_ID))
        .expect("one golden case carries the <think> token");
    assert!(
        think_case.text.contains("<think>"),
        "the golden case carrying id {THINK_TOKEN_ID} must be the one whose raw text contains \
         the literal <think> tag"
    );

    record_executed(Capability::Bonsai2Models, TEST);
}

/// The golden fixture itself: every case must be non-empty and the
/// `<think>` id must appear in exactly the case that names it in its raw
/// text — a fixture-authoring mistake here would make the real-model test
/// above pass or fail for the wrong reason.
#[test]
fn golden_tokenizations_fixture_is_well_formed_bonsai2() {
    for case in &GOLDEN_TOKENIZATIONS {
        assert!(!case.text.is_empty(), "golden case has empty text");
        assert!(
            !case.ids.is_empty(),
            "{:?}: golden case has no token ids",
            case.text
        );
    }
    let think_cases: Vec<_> = GOLDEN_TOKENIZATIONS
        .iter()
        .filter(|c| c.ids.contains(&THINK_TOKEN_ID))
        .collect();
    assert_eq!(
        think_cases.len(),
        1,
        "expected exactly one golden case to carry the <think> id {THINK_TOKEN_ID}"
    );
}

// ─────────────────────────────────────────────────────────────────────────
//  G7: chat-template rendering
// ─────────────────────────────────────────────────────────────────────────

/// One golden `apply-template` case: the messages/options to render and the
/// fork's own exact `output.prompt` string — the fork's `/apply-template`
/// dump, embedded verbatim (5 cases: plain, `enable_thinking=false`,
/// `reasoning_effort=low`, a multi-turn history with a past turn's
/// `reasoning_content`, and a tool-call schema). Every case's `add_generation_prompt`
/// is `true` (the fork's endpoint default whenever the field is omitted, as
/// it is in every one of these captures): each golden `prompt` ends with the
/// assistant-turn opener.
struct GoldenTemplateCase {
    tools: Option<&'static str>,
    enable_thinking: Option<bool>,
    reasoning_effort: Option<&'static str>,
    prompt: &'static str,
}

fn msg(role: &'static str, content: &'static str) -> RenderMessage {
    RenderMessage::new(role, content)
}

fn golden_template_cases() -> Vec<GoldenTemplateCase> {
    vec![
        GoldenTemplateCase {
            tools: None,
            enable_thinking: None,
            reasoning_effort: None,
            prompt: "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully \
                     through the task, validate key assumptions, consider plausible alternatives, \
                     and prioritize correctness, consistency, and clarity in the final answer.\
                     <|im_end|>\n<|im_start|>user\nWhat is 2+2? Answer briefly.<|im_end|>\n\
                     <|im_start|>assistant\n<think>\n",
        },
        GoldenTemplateCase {
            tools: None,
            enable_thinking: Some(false),
            reasoning_effort: None,
            prompt: "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n\n\
                     </think>\n\n",
        },
        GoldenTemplateCase {
            tools: None,
            enable_thinking: None,
            reasoning_effort: Some("low"),
            prompt: "<|im_start|>system\nReasoning effort is set to low. Keep your thinking brief \
                     and focused, moving directly to the conclusion without unnecessary \
                     elaboration.<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n\
                     <|im_start|>assistant\n<think>\n",
        },
        GoldenTemplateCase {
            tools: None,
            enable_thinking: None,
            reasoning_effort: None,
            prompt: "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully \
                     through the task, validate key assumptions, consider plausible alternatives, \
                     and prioritize correctness, consistency, and clarity in the final answer.\n\n\
                     You are a helpful assistant<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n\
                     <|im_start|>assistant\n<think>\nuser greets\n</think>\n\nHello!<|im_end|>\n\
                     <|im_start|>user\nBye<|im_end|>\n<|im_start|>assistant\n<think>\n",
        },
        GoldenTemplateCase {
            tools: Some(
                r#"[{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]"#,
            ),
            enable_thinking: None,
            reasoning_effort: None,
            prompt: "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully \
                     through the task, validate key assumptions, consider plausible alternatives, \
                     and prioritize correctness, consistency, and clarity in the final answer.\n\n\
                     # Tools\n\nYou have access to the following functions:\n\n<tools>\n{\"type\": \
                     \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \
                     \"Get weather\", \"parameters\": {\"type\": \"object\", \"properties\": \
                     {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}}\n</tools>\n\n\
                     If you choose to call a function ONLY reply in the following format with NO \
                     suffix:\n\n<tool_call>\n<function=example_function_name>\n\
                     <parameter=example_parameter_1>\nvalue_1\n</parameter>\n\
                     <parameter=example_parameter_2>\nThis is the value for the second parameter\n\
                     that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n\
                     <IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an \
                     inner <function=...></function> block must be nested within \
                     <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- \
                     You may provide optional reasoning for your function call in natural language \
                     BEFORE the function call, but NOT after\n- If there is no function call \
                     available, answer the question like normal with your current knowledge and do \
                     not tell the user about function calls\n</IMPORTANT><|im_end|>\n\
                     <|im_start|>user\nweather?<|im_end|>\n<|im_start|>assistant\n<think>\n",
        },
    ]
}

#[test]
fn real_27b_chat_template_matches_the_fork_for_all_five_golden_cases_bonsai2() {
    const TEST: &str = "oxibonsai-runtime::bonsai2_runtime_tests::\
                        real_27b_chat_template_matches_the_fork_for_all_five_golden_cases_bonsai2";
    let Some(path) = locate_27b_gguf(TEST) else {
        return;
    };

    let mmap = mmap_gguf_file(&path).unwrap_or_else(|e| panic!("mmap {}: {e}", path.display()));
    let gguf = GgufFile::parse(&mmap).expect("real 27B GGUF header parses");
    let template = ResolvedChatTemplate::from_gguf(&gguf.metadata)
        .expect("GGUF-embedded chat template compiles");

    // The golden messages, built once here (rather than stored `'static` in
    // `GoldenTemplateCase`, since `RenderMessage` owns `String`s) and paired
    // with each case by index — case order matches `golden_template_cases`'s
    // own five, one to one.
    let messages_per_case: Vec<Vec<RenderMessage>> = vec![
        vec![msg("user", "What is 2+2? Answer briefly.")],
        vec![msg("user", "What is 2+2?")],
        vec![msg("user", "What is 2+2?")],
        vec![
            msg("system", "You are a helpful assistant"),
            msg("user", "Hi"),
            msg("assistant", "Hello!").with_reasoning_content("user greets"),
            msg("user", "Bye"),
        ],
        vec![msg("user", "weather?")],
    ];

    let cases = golden_template_cases();
    assert_eq!(cases.len(), messages_per_case.len());

    let mut failures = Vec::new();
    for (case, messages) in cases.iter().zip(&messages_per_case) {
        let opts = RenderOptions {
            add_generation_prompt: true,
            enable_thinking: case.enable_thinking,
            reasoning_effort: case.reasoning_effort.map(str::to_string),
            preserve_thinking: None,
            add_vision_id: false,
            tools: case.tools.map(str::to_string),
        };
        match template.render_with(messages, &opts) {
            Ok(rendered) if rendered == case.prompt => {
                eprintln!("G7 case ok: {:?}", &rendered[..rendered.len().min(60)]);
            }
            Ok(rendered) => failures.push(format!(
                "rendered prompt differs from the fork's:\n  ours: {rendered:?}\n  fork: {:?}",
                case.prompt
            )),
            Err(e) => failures.push(format!("render_with failed: {e}")),
        }
    }
    assert!(
        failures.is_empty(),
        "real 27B chat-template rendering (G7) diverged from the fork:\n{}",
        failures.join("\n---\n")
    );

    record_executed(Capability::Bonsai2Models, TEST);
}

// ─────────────────────────────────────────────────────────────────────────
//  G8: reasoning-content splitting
// ─────────────────────────────────────────────────────────────────────────

/// `</think>`'s vocabulary id — [`THINK_TOKEN_ID`] above is the opener.
const REASONING_CLOSE_TOKEN_ID: u32 = 248069;

/// G8: [`oxibonsai_runtime::reasoning`]'s streaming splitter against a
/// vendored capture of the fork's own `/v1/chat/completions` response for
/// prompt 1 (`tests/fixtures/bonsai2_golden/chat.prompt1.server.json`, a
/// trimmed copy keeping only the final message split and the raw per-token
/// generation stream). Needs no real GGUF — the splitter is pure per-token
/// logic — so this runs unconditionally, in every plain `cargo test`/`cargo
/// nextest run`, rather than self-skipping on a missing model file.
///
/// The capture's `token_stream` is the model's raw per-token generation
/// output, the same shape the server's own streaming response path decodes
/// one token at a time; this test drives [`ReasoningSplitter::push`] over it
/// exactly that way, then asserts the accumulated `reasoning_content`/
/// `content` against the capture's own final message fields — checking this
/// crate's splitter against the reference server's actual output, not only
/// against hand-written unit fixtures.
///
/// Bonsai 2's chat template opens `<think>` itself (the prompt already ends
/// `<|im_start|>assistant\n<think>\n`), so the raw output stream never
/// repeats that opening tag — only the closing `</think>` appears — which is
/// why this test constructs the splitter with `started_in_think = true` and
/// no model-emitted `open_id`. The stream's first token after the close
/// boundary is `"\n\n"` (the template's own historical re-rendering
/// convention, `'\n</think>\n\n' + content`); the splitter swallows that
/// whole leading-newline run once, which is why `content` below is the bare
/// `"4"` even though the raw stream carries the extra newlines.
#[test]
fn reasoning_split_matches_the_fork_capture_for_prompt_one_bonsai2() {
    let raw = include_str!("fixtures/bonsai2_golden/chat.prompt1.server.json");
    let capture: serde_json::Value =
        serde_json::from_str(raw).expect("vendored G8 fixture is well-formed JSON");

    let expected_content = capture["message"]["content"]
        .as_str()
        .expect("fixture message.content is a string");
    let expected_reasoning = capture["message"]["reasoning_content"]
        .as_str()
        .expect("fixture message.reasoning_content is a string");

    let stream = capture["token_stream"]
        .as_array()
        .expect("fixture token_stream is an array");
    assert!(!stream.is_empty(), "fixture token_stream must not be empty");
    let tokens: Vec<(u32, &str)> = stream
        .iter()
        .map(|entry| {
            let id = entry["id"].as_u64().expect("token_stream entry.id") as u32;
            let piece = entry["token"].as_str().expect("token_stream entry.token");
            (id, piece)
        })
        .collect();

    // Drive the streaming splitter one token at a time, exactly as the
    // server's own streaming response path does, rather than only calling
    // the batch convenience wrapper.
    let mut splitter = ReasoningSplitter::new(true, Some(REASONING_CLOSE_TOKEN_ID));
    assert!(
        splitter.in_reasoning(),
        "Bonsai 2's chat template opens <think> itself, so the splitter must \
         start in the reasoning phase"
    );
    let mut streamed_reasoning = String::new();
    let mut streamed_content = String::new();
    let mut saw_boundary = false;
    for &(id, piece) in &tokens {
        match splitter.push(id, piece) {
            ReasoningChunk::Reasoning(s) => streamed_reasoning.push_str(&s),
            ReasoningChunk::Content(s) => streamed_content.push_str(&s),
            ReasoningChunk::Boundary => saw_boundary = true,
        }
    }
    assert!(
        saw_boundary,
        "the fixture's token stream must carry the </think> close id \
         ({REASONING_CLOSE_TOKEN_ID}) for this gate to mean anything"
    );
    assert!(
        !splitter.in_reasoning(),
        "the splitter must have left the reasoning phase by the end of the stream"
    );
    assert_eq!(
        streamed_reasoning, expected_reasoning,
        "streamed reasoning_content diverged from the fork's own capture"
    );
    assert_eq!(
        streamed_content, expected_content,
        "streamed content diverged from the fork's own capture"
    );

    // Cross-check against the non-streaming, one-call convenience wrapper:
    // both paths drive the same underlying `ReasoningSplitter` state machine
    // and must agree.
    let (batch_reasoning, batch_content) =
        split_reasoning(tokens, true, Some(REASONING_CLOSE_TOKEN_ID));
    assert_eq!(
        batch_reasoning.as_deref(),
        Some(expected_reasoning),
        "split_reasoning's reasoning_content diverged from the streaming result"
    );
    assert_eq!(
        batch_content, expected_content,
        "split_reasoning's content diverged from the streaming result"
    );
}

// ─────────────────────────────────────────────────────────────────────────
//  G9: `oxibonsai info` on the real 27B files
// ─────────────────────────────────────────────────────────────────────────

/// `oxibonsai info`'s own printed variant/quant-id/layer-split report lives
/// entirely inside the `oxibonsai-cli` binary crate (`src/cli/cmd_info.rs`,
/// `src/cli/model_desc.rs`) — a bin-only crate with no `[lib]` target, so no
/// other crate's test can import its modules directly. This exercises the
/// real, shipped `oxibonsai info --json` command as a subprocess instead:
/// build the binary (a no-op once it is up to date for this exact feature
/// set — but a rebuild, not a no-op, if another test most recently built it
/// with a different one; `crates/oxibonsai-runtime/tests/legacy_parity_tests.rs`'s
/// CLI-CORE case builds the same binary with `--features eval`, which this
/// call does not pass), run it against the real GGUF, and parse its own
/// `--json` output, exactly as an operator running `oxibonsai info --model
/// <file> --json` would see it.
fn target_dir() -> std::path::PathBuf {
    if let Ok(dir) = std::env::var("CARGO_TARGET_DIR") {
        if !dir.is_empty() {
            return std::path::PathBuf::from(dir);
        }
    }
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target")
}

/// Run `oxibonsai info --json --model <path>` and parse its stdout.
///
/// # Panics
/// If the binary fails to build, exits non-zero, or its stdout is not valid
/// JSON — every one of those is a genuine gate failure, not a skip (the
/// caller has already confirmed the model file exists before calling this).
fn run_oxibonsai_info_json(model_path: &std::path::Path) -> serde_json::Value {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let build_status = std::process::Command::new(&cargo)
        .args(["build", "--release", "-p", "oxibonsai-cli", "--bin", "oxibonsai"])
        .status()
        .expect("spawning `cargo build -p oxibonsai-cli --bin oxibonsai` should not itself fail to launch");
    assert!(
        build_status.success(),
        "cargo build -p oxibonsai-cli --bin oxibonsai failed (exit {:?})",
        build_status.code()
    );

    let bin = target_dir().join("release").join("oxibonsai");
    let output = std::process::Command::new(&bin)
        .args(["info", "--json", "--model"])
        .arg(model_path)
        .output()
        .unwrap_or_else(|e| panic!("spawning {} failed: {e}", bin.display()));
    assert!(
        output.status.success(),
        "{} info --json --model {} exited with {:?}\nstdout:\n{}\nstderr:\n{}",
        bin.display(),
        model_path.display(),
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap_or_else(|e| {
        panic!(
            "oxibonsai info --json produced non-JSON stdout: {e}\nstdout:\n{}",
            String::from_utf8_lossy(&output.stdout)
        )
    })
}

/// Asserts `oxibonsai info --json`'s report of `path` matches the design's
/// facts for a Bonsai 2 27B hybrid: 64 total layers split 16 full-attention
/// / 48 Gated-DeltaNet, and the given ggml quantization type id (PQ2_0 =
/// 142, PTQ1_0 = 143) with a variant name that names both the model family
/// and the quant band.
fn assert_info_reports_hybrid_facts(
    path: &std::path::Path,
    expected_ggml_type_id: u64,
    expected_variant_substring: &str,
) {
    let info = run_oxibonsai_info_json(path);
    let hybrid = &info["hybrid"];
    assert!(
        !hybrid.is_null(),
        "oxibonsai info --json reported no \"hybrid\" section for a qwen35 file: {info}"
    );

    let variant = info["variant"]
        .as_str()
        .unwrap_or_else(|| panic!("oxibonsai info --json reported no \"variant\": {info}"));
    assert!(
        variant.contains("27B") && variant.contains(expected_variant_substring),
        "variant {variant:?} does not name a Bonsai 2 27B {expected_variant_substring} model"
    );

    assert_eq!(hybrid["layers"].as_u64(), Some(64), "hybrid.layers: {info}");
    let full_attention_layers = hybrid["full_attention_layers"]
        .as_array()
        .unwrap_or_else(|| panic!("hybrid.full_attention_layers is not an array: {info}"));
    assert_eq!(
        full_attention_layers.len(),
        16,
        "expected 16 full-attention layers: {info}"
    );
    assert_eq!(
        hybrid["gated_deltanet_layers"].as_u64(),
        Some(48),
        "hybrid.gated_deltanet_layers: {info}"
    );

    let ggml_type_id = hybrid["weights"]["ggml_type_id"].as_u64();
    assert_eq!(
        ggml_type_id,
        Some(expected_ggml_type_id),
        "hybrid.weights.ggml_type_id: {info}"
    );
}

/// G9 over the PQ2_0 band (ggml type id 142).
#[test]
fn real_27b_info_reports_pq2_0_variant_and_layer_split_bonsai2() {
    const TEST: &str =
        "oxibonsai-runtime::bonsai2_runtime_tests::real_27b_info_reports_pq2_0_variant_and_layer_split_bonsai2";
    let Some(path) = locate_named_27b_gguf("OXI_BONSAI2_PQ2_GGUF", TEST) else {
        return;
    };
    assert_info_reports_hybrid_facts(&path, 142, "PQ2_0");
    record_executed(Capability::Bonsai2Models, TEST);
}

/// G9 over the PTQ1_0 band (ggml type id 143).
#[test]
fn real_27b_info_reports_ptq1_0_variant_and_layer_split_bonsai2() {
    const TEST: &str =
        "oxibonsai-runtime::bonsai2_runtime_tests::real_27b_info_reports_ptq1_0_variant_and_layer_split_bonsai2";
    let Some(path) = locate_named_27b_gguf("OXI_BONSAI2_PTQ1_GGUF", TEST) else {
        return;
    };
    assert_info_reports_hybrid_facts(&path, 143, "PTQ1_0");
    record_executed(Capability::Bonsai2Models, TEST);
}

/// Locates one Bonsai 2 27B file by its own env var only (no `models/`
/// fallback: unlike G6/G7 above, this gate spawns a `cargo build` and a
/// fresh process per case rather than only `mmap`ping a header, so it must
/// never run by accident just because a `models/` directory happens to be
/// populated — the same env-only convention `hybrid_metal_gates.rs` uses).
/// Records a capability skip (naming `test_name`) when the file is absent.
fn locate_named_27b_gguf(env_var: &str, test_name: &str) -> Option<std::path::PathBuf> {
    let require_real_files = std::env::var("OXI_REQUIRE_MODEL_FILES")
        .map(|v| v == "1")
        .unwrap_or(false);
    if let Ok(path) = std::env::var(env_var) {
        if !path.is_empty() {
            let path = std::path::PathBuf::from(path);
            if path.is_file() {
                return Some(path);
            }
        }
    }
    assert!(
        !require_real_files,
        "OXI_REQUIRE_MODEL_FILES=1: {test_name} needs a real Bonsai 2 27B GGUF; set {env_var}"
    );
    eprintln!("skip {test_name}: {env_var} not set to an existing file");
    record_skipped(Capability::Bonsai2Models, test_name);
    None
}
