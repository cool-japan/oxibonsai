//! Unit tests for `cmd_run.rs` (sibling file, declared there via `#[path]`,
//! so `super` still names that module).

use super::*;
use crate::cli::test_fixtures::{self, tiny_dense_gguf};
use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue};
use oxibonsai_runtime::config::RopeScalingMode;
use oxibonsai_runtime::engine_seam::Backend;

// ── --chat contract flags ───────────────────────────────────────────────────

#[test]
fn contract_flags_without_chat_are_refused_naming_each_flag() {
    let contract = ChatContract {
        enable_thinking: Some(false),
        reasoning_effort: Some("low".to_string()),
        tools_json: Some("[]".to_string()),
    };
    let msg = require_chat_for_contract_flags(false, &contract, true)
        .expect_err("a raw run cannot honour them")
        .to_string();
    for flag in [
        "--no-think",
        "--reasoning-effort",
        "--tools",
        "--show-reasoning/--hide-reasoning",
        "--chat",
    ] {
        assert!(msg.contains(flag), "names {flag}: {msg}");
    }
    let think = ChatContract {
        enable_thinking: Some(true),
        ..ChatContract::default()
    };
    let msg = require_chat_for_contract_flags(false, &think, false)
        .expect_err("--think without --chat")
        .to_string();
    assert!(msg.contains("--think"), "{msg}");
}

#[test]
fn contract_flags_with_chat_or_no_flags_at_all_are_accepted() {
    let contract = ChatContract {
        enable_thinking: Some(true),
        ..ChatContract::default()
    };
    require_chat_for_contract_flags(true, &contract, true).expect("--chat renders them");
    require_chat_for_contract_flags(false, &ChatContract::default(), false)
        .expect("a plain raw run");
}

// ── RT-17: the three sampling precedences ───────────────────────────────────

fn metadata_gguf(entries: Vec<(&str, MetadataWriteValue)>) -> Vec<u8> {
    let mut w = GgufWriter::new();
    w.add_metadata(
        "general.architecture",
        MetadataWriteValue::Str("qwen35".to_string()),
    );
    for (key, value) in entries {
        w.add_metadata(key, value);
    }
    w.to_bytes().expect("serialize")
}

/// The real Bonsai 2 declaration: `temp` 1.0, `top_p` 0.95, `top_k` 20 as
/// GGUF INT32 (the on-disk type), plus a `min_p`.
fn bonsai2_sampling_gguf() -> Vec<u8> {
    metadata_gguf(vec![
        ("general.sampling.temp", MetadataWriteValue::F32(1.0)),
        ("general.sampling.top_p", MetadataWriteValue::F32(0.95)),
        ("general.sampling.top_k", MetadataWriteValue::I32(20)),
        ("general.sampling.min_p", MetadataWriteValue::F32(0.05)),
    ])
}

#[test]
fn an_explicit_value_beats_the_gguf_default() {
    let bytes = bonsai2_sampling_gguf();
    let gguf = GgufFile::parse(&bytes).expect("parse");
    let resolved =
        resolve_sampling(Some(0.3), Some(7), Some(0.5), Some(0.2), &gguf.metadata).expect("valid");
    assert_eq!(
        resolved,
        ResolvedSampling {
            temperature: 0.3,
            top_k: 7,
            top_p: 0.5,
            min_p: 0.2
        }
    );
}

#[test]
fn the_gguf_default_beats_the_literal_when_nothing_is_explicit() {
    let bytes = bonsai2_sampling_gguf();
    let gguf = GgufFile::parse(&bytes).expect("parse");
    let resolved = resolve_sampling(None, None, None, None, &gguf.metadata).expect("valid");
    assert_eq!(
        resolved,
        ResolvedSampling {
            temperature: 1.0,
            top_k: 20,
            top_p: 0.95,
            min_p: 0.05
        }
    );
    // Precedence is per field: an explicit temperature alone still leaves
    // the other three to the model.
    let mixed = resolve_sampling(Some(0.0), None, None, None, &gguf.metadata).expect("valid");
    assert_eq!(mixed.temperature, 0.0);
    assert_eq!((mixed.top_k, mixed.top_p, mixed.min_p), (20, 0.95, 0.05));
}

#[test]
fn the_literal_applies_when_neither_is_declared() {
    let bytes = metadata_gguf(Vec::new());
    let gguf = GgufFile::parse(&bytes).expect("parse");
    let resolved = resolve_sampling(None, None, None, None, &gguf.metadata).expect("valid");
    assert_eq!(
        resolved,
        ResolvedSampling {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            min_p: 0.0
        }
    );
}

#[test]
fn an_out_of_range_gguf_default_is_refused_like_a_flag() {
    let bytes = metadata_gguf(vec![(
        "general.sampling.top_p",
        MetadataWriteValue::F32(5.0),
    )]);
    let gguf = GgufFile::parse(&bytes).expect("parse");
    let err = resolve_sampling(None, None, None, None, &gguf.metadata).expect_err("top_p 5.0");
    assert!(err.to_string().contains("top-p"), "{err}");
    // ...unless an explicit value overrides the bad declaration.
    resolve_sampling(None, None, Some(0.9), None, &gguf.metadata).expect("overridden");
}

// ── load-time guards: default context, ctx guard, --rope-scaling pre-flight ──

#[test]
fn the_default_context_is_per_architecture() {
    let dense = tiny_dense_gguf(Vec::new());
    let gguf = GgufFile::parse(&dense).expect("parse");
    let ctx = apply_bonsai2_load_time_guards(
        &gguf,
        "qwen3",
        dense.len() as u64,
        None,
        RopeScalingMode::Auto,
    )
    .expect("dense default");
    assert_eq!(ctx, 4096);
    let ctx = apply_bonsai2_load_time_guards(
        &gguf,
        "qwen3",
        dense.len() as u64,
        Some(123),
        RopeScalingMode::Auto,
    )
    .expect("explicit");
    assert_eq!(ctx, 123);

    let mut w = GgufWriter::new();
    test_fixtures::qwen35_27b_metadata(&mut w);
    let hybrid = w.to_bytes().expect("serialize");
    let gguf = GgufFile::parse(&hybrid).expect("parse");
    let ctx = apply_bonsai2_load_time_guards(
        &gguf,
        "qwen35",
        hybrid.len() as u64,
        None,
        RopeScalingMode::Auto,
    )
    .expect("qwen35 default");
    assert_eq!(ctx, 8192);
    let err = apply_bonsai2_load_time_guards(
        &gguf,
        "qwen35",
        hybrid.len() as u64,
        Some(300_000),
        RopeScalingMode::Auto,
    )
    .expect_err("--ctx 300000");
    let msg = err.to_string();
    assert!(msg.contains("300000") && msg.contains("262144"), "{msg}");
}

#[test]
fn rope_scaling_on_is_refused_before_loading_a_file_that_declares_none() {
    let bytes = tiny_dense_gguf(Vec::new());
    let gguf = GgufFile::parse(&bytes).expect("parse");
    let err = apply_bonsai2_load_time_guards(
        &gguf,
        "qwen3",
        bytes.len() as u64,
        None,
        RopeScalingMode::On,
    )
    .expect_err("nothing to force on");
    let msg = err.to_string();
    assert!(msg.contains("--rope-scaling on"), "{msg}");
    assert!(msg.contains("rope.scaling"), "{msg}");
    for mode in [RopeScalingMode::Auto, RopeScalingMode::Off] {
        apply_bonsai2_load_time_guards(&gguf, "qwen3", bytes.len() as u64, None, mode)
            .expect("auto/off never refuse");
    }
}

// ── P0: `--backend cpu --temperature 0` never takes a GPU route ─────────────

fn greedy_load(backend: Backend, seed: u64, temperature: f32) -> EngineLoad {
    EngineLoad {
        params: build_sampling_params(temperature, 40, 0.9, 1.0),
        seed,
        max_seq_len: 64,
        backend,
        rope_scaling: RopeScalingMode::Auto,
        prefill_chunk: None,
        penalties: PenaltyParams::default(),
    }
}

/// The orchestrator's P0: the CLI's own temperature-0 shortcut decoded a
/// CPU-tier engine through the GPU argmax path. The shortcut is gone; the
/// engine alone decides, and a `--backend cpu` engine is never eligible —
/// on every build, `metal` included. The streamed, the `--no-stream` and a
/// fresh engine's own `generate` all agree token for token.
#[test]
fn a_cpu_backend_greedy_run_never_takes_the_gpu_argmax_route() {
    let bytes = tiny_dense_gguf(Vec::new());
    let gguf = GgufFile::parse(&bytes).expect("parse");
    let prompt = [1u32, 2, 3];

    let mut engine = load_engine(&gguf, &greedy_load(Backend::Cpu, 42, 0.0), 0).expect("load");
    assert!(
        !engine.greedy_gpu_eligible(false),
        "a --backend cpu engine must never be GPU-argmax eligible"
    );
    let mut printer = TokenPrinter::new(None, false, ReasoningDisplay::Show, true);
    let streamed = run_engine_generation(&mut engine, &prompt, 8, false, &mut printer)
        .expect("streamed decode");
    assert!(matches!(printer, TokenPrinter::Raw { count } if count == streamed));

    let mut engine = load_engine(&gguf, &greedy_load(Backend::Cpu, 42, 0.0), 0).expect("load");
    let mut printer = TokenPrinter::new(None, false, ReasoningDisplay::Show, true);
    let buffered = run_engine_generation(&mut engine, &prompt, 8, true, &mut printer)
        .expect("--no-stream decode");
    assert_eq!(streamed, buffered);

    let mut engine = load_engine(&gguf, &greedy_load(Backend::Cpu, 42, 0.0), 0).expect("load");
    let reference = engine.generate(&prompt, 8).expect("engine generate");
    assert_eq!(
        reference.len(),
        streamed,
        "the CLI path emits exactly the engine's tokens"
    );
}

/// cli-09: nothing in the CLI forces a GPU tier. A default-feature build
/// (no `gpu`) always resolves `--backend auto` to a CPU tier; a `gpu`
/// build may pick the GPU for a dense model, but `--backend cpu` is CPU on
/// every build.
#[test]
fn the_default_build_resolves_to_a_cpu_kernel_tier() {
    let bytes = tiny_dense_gguf(Vec::new());
    let gguf = GgufFile::parse(&bytes).expect("parse");
    let cpu = load_engine(&gguf, &greedy_load(Backend::Cpu, 42, 0.0), 0).expect("load");
    assert!(!cpu.greedy_gpu_eligible(false));
    assert!(!cpu.uses_fused_gpu_decode());
    if !cfg!(feature = "gpu") {
        let auto = load_engine(&gguf, &greedy_load(Backend::Auto, 42, 0.0), 0).expect("load");
        assert!(
            !auto.uses_fused_gpu_decode() && !auto.greedy_gpu_eligible(false),
            "a default-feature build must never select a GPU tier"
        );
        let label = auto.kernel_label();
        assert!(
            !label.to_ascii_lowercase().contains("metal")
                && !label.to_ascii_lowercase().contains("cuda"),
            "default build kernel label: {label}"
        );
    }
}

// ── RT-12: --seed makes a sampled CLI run reproducible ──────────────────────

fn sampled_tokens(bytes: &[u8], seed: u64, min_p: f32) -> Vec<u32> {
    let gguf = GgufFile::parse(bytes).expect("parse");
    let load = greedy_load(Backend::Cpu, seed, 0.8);
    let mut engine = load_engine(&gguf, &load, 0).expect("load");
    let prompt = [1u32, 2, 3];
    let mut out = Vec::new();
    if generate::needs_cli_sampler(0.8, min_p) {
        let mut sampler = session_sampler(&load, min_p);
        generate::decode_with_sampler(&mut engine, &prompt, 16, &mut sampler, |token| {
            out.push(token);
            Ok(true)
        })
        .expect("min-p decode");
    } else {
        let (tx, rx) = std::sync::mpsc::channel::<u32>();
        engine
            .generate_streaming_sync(&prompt, 16, &tx)
            .expect("engine decode");
        drop(tx);
        out.extend(rx);
    }
    out
}

#[test]
fn the_same_seed_reproduces_a_sampled_run_and_another_seed_does_not() {
    // Every logit of the fixture ties, so temperature 0.8 samples
    // (near-)uniformly: two seeds agreeing on 16 draws is ~29^-16.
    let bytes = tiny_dense_gguf(Vec::new());
    for min_p in [0.0f32, 0.05] {
        let first = sampled_tokens(&bytes, 7, min_p);
        assert_eq!(first.len(), 16, "min_p {min_p}");
        assert_eq!(
            first,
            sampled_tokens(&bytes, 7, min_p),
            "seed 7 twice (min_p {min_p})"
        );
        assert_ne!(
            first,
            sampled_tokens(&bytes, 8, min_p),
            "seed 7 vs 8 (min_p {min_p})"
        );
    }
}

// ── Real models (env-gated, one at a time) ──────────────────────────────────

/// Greedy-decode `prompt` for `max_tokens` through the CLI's own path
/// (`load_engine` + the production tokenizer resolution + the engine loop +
/// the buffered printer) and return the printed continuation.
fn cli_greedy_text(
    model: &std::path::Path,
    prompt: &str,
    max_tokens: usize,
    backend: Backend,
    rope_scaling: RopeScalingMode,
) -> String {
    let mmap = oxibonsai_core::gguf::reader::mmap_gguf_file(model).expect("mmap");
    let gguf = GgufFile::parse(&mmap).expect("parse");
    let model_str = model.to_string_lossy().into_owned();
    let load = EngineLoad {
        rope_scaling,
        max_seq_len: 256,
        ..greedy_load(backend, 42, 0.0)
    };
    let mut engine = load_engine(&gguf, &load, 0).expect("load");
    let expected_vocab = model_vocab_size(&gguf).ok();
    let lookup = resolve_tokenizer_vocab_aware(None, &model_str, expected_vocab);
    let tok = resolve_model_tokenizer(
        None,
        &lookup,
        &gguf,
        expected_vocab,
        TokenizerBackendChoice::Auto,
        false,
    )
    .expect("tokenizer resolution")
    .expect("a tokenizer for the real model");
    let prompt_tokens = tok.encode(prompt).expect("encode");
    let mut printer = TokenPrinter::new(Some(&tok), false, ReasoningDisplay::Show, false);
    run_engine_generation(&mut engine, &prompt_tokens, max_tokens, false, &mut printer)
        .expect("decode");
    printer.finish(None).1
}

/// The P0's own acceptance, through the CLI path: `--backend cpu
/// --temperature 0` on the legacy 1.7B is pure greedy and reproduces the
/// greedy truth — `golden_legacy/legacy_golden.json`'s backend "metal" row
/// for prompt 1 (the orchestrator P0 addendum's chosen truth), i.e. CPU and
/// Metal are byte-identical. That file's backend "cpu" row for this prompt
/// (`"... Tokyo.\nThe answer to the question ..."`) is a capture of the
/// pre-P0 CLI, whose CPU path applied a hidden `repetition_penalty` of 1.1
/// before the argmax (`findings/legacy_cpu_metal_divergence.md`); reproducing
/// it would mean the hidden penalty is back.
#[test]
fn a_cpu_greedy_run_reproduces_the_legacy_1_7b_greedy_golden() {
    let Some(model) = test_fixtures::models_dir_file("Ternary-Bonsai-1.7B.gguf") else {
        return;
    };
    let _real = test_fixtures::real_model_lock();
    let text = cli_greedy_text(
        &model,
        "The capital of Japan is",
        32,
        Backend::Cpu,
        RopeScalingMode::Auto,
    );
    assert_eq!(
        text,
        " Tokyo. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The capital of \
         Japan is Tokyo. The capital of Japan is Tokyo. The capital"
    );
}

/// Ruling R-ROPEOFF: `--rope-scaling off` reproduces the pre-YaRN
/// Bonsai-8B prompt-3 text (the pre-session anchor), while `auto` (the
/// default) reproduces the PrismML fork's YaRN text.
#[test]
fn rope_scaling_off_reproduces_the_pre_yarn_bonsai_8b_text() {
    let Some(model) = test_fixtures::models_dir_file("Bonsai-8B.gguf") else {
        return;
    };
    let _real = test_fixtures::real_model_lock();
    let prompt = "Once upon a time, in a small village by the sea,";
    let off = cli_greedy_text(&model, prompt, 32, Backend::Cpu, RopeScalingMode::Off);
    assert_eq!(
        off,
        " there lived a young girl named Lila. She was known for her love of the sea and her \
         ability to speak with the waves. One day, while exploring"
    );
    let auto = cli_greedy_text(&model, prompt, 32, Backend::Cpu, RopeScalingMode::Auto);
    assert_eq!(
        auto,
        " there lived a young girl named Lila. She was known for her curious nature and her \
         love of the sea. Lila often spent her days exploring the cliffs"
    );
}
