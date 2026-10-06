//! Tests for `--prefill-chunk` ([`apply_prefill_chunk`]): sibling of
//! `bonsai2_tests.rs`, declared in `bonsai2.rs` via `#[path]`, so `super`
//! still names that module.

use super::*;
use oxibonsai_core::gguf::reader::GgufFile;

// ── `--prefill-chunk`: the chunk the model holds is the one reported ────────

/// A model that clamps a request to `max` and rounds it down to a multiple of
/// `multiple`, standing in for an engine that does not hold exactly what it
/// was asked for.
struct AdjustingModel {
    held: usize,
    max: usize,
    multiple: usize,
}

impl PrefillChunkTarget for AdjustingModel {
    fn kind(&self) -> PrefillChunkKind {
        PrefillChunkKind::Hybrid
    }

    fn set_chunk(&mut self, chunk: usize) -> anyhow::Result<()> {
        anyhow::ensure!(chunk > 0, "--prefill-chunk 0: a zero chunk");
        self.held = (chunk.min(self.max) / self.multiple * self.multiple).max(self.multiple);
        Ok(())
    }

    fn held_chunk(&self) -> usize {
        self.held
    }
}

/// A request the model clamps or rounds reports the value the model holds,
/// with the request beside it; one it takes as given reports one number.
#[test]
fn a_clamped_or_rounded_request_reports_the_chunk_the_model_holds() {
    let mut model = AdjustingModel {
        held: 512,
        max: 4096,
        multiple: 32,
    };

    // Clamped: asked for 100000, holds 4096.
    let clamped = set_prefill_chunk_on(&mut model, 100_000).expect("set");
    assert_eq!(
        clamped,
        PrefillChunkOutcome {
            kind: PrefillChunkKind::Hybrid,
            requested: 100_000,
            honoured: 4096,
            in_effect: 4096,
        },
        "the honoured value is read back from the model, not echoed from the flag"
    );
    assert!(!clamped.capped_by_executor());
    let line = clamped.message();
    assert!(line.contains("is 4096 tokens"), "{line}");
    assert!(line.contains("asked for 100000"), "{line}");
    assert!(!line.contains("set to 100000"), "{line}");

    // Rounded: asked for 100, holds 96.
    let rounded = set_prefill_chunk_on(&mut model, 100).expect("set");
    assert_eq!((rounded.requested, rounded.honoured), (100, 96));
    let line = rounded.message();
    assert!(line.contains("is 96 tokens"), "{line}");
    assert!(line.contains("asked for 100"), "{line}");

    // Taken as given: one number, no "asked for".
    let exact = set_prefill_chunk_on(&mut model, 64).expect("set");
    assert_eq!((exact.requested, exact.honoured), (64, 64));
    let line = exact.message();
    assert_eq!(line, "hybrid Gated-DeltaNet prefill chunk set to 64 tokens");
    assert!(!line.contains("asked for"), "{line}");

    // The model's own refusal is the caller's error.
    let refused = set_prefill_chunk_on(&mut model, 0).expect_err("a zero chunk");
    assert!(
        refused.to_string().contains("--prefill-chunk 0"),
        "{refused}"
    );
}

#[test]
fn the_prefill_chunk_line_names_which_model_it_set() {
    let hybrid = PrefillChunkOutcome {
        kind: PrefillChunkKind::Hybrid,
        requested: 64,
        honoured: 64,
        in_effect: 64,
    };
    let dense = PrefillChunkOutcome {
        kind: PrefillChunkKind::Dense,
        requested: 64,
        honoured: 48,
        in_effect: 48,
    };
    assert_eq!(
        hybrid.message(),
        "hybrid Gated-DeltaNet prefill chunk set to 64 tokens"
    );
    assert_eq!(
        dense.message(),
        "dense chunked-prefill size is 48 tokens (--prefill-chunk asked for 64; the model \
         adjusted it)"
    );
}

/// An executor whose KV-window memory budget holds its prefill calls below
/// the chunk the model holds: the line reports the call size in effect, the
/// request beside it and why — never "set to" the request — and the outcome
/// says it was capped (the line is then a `WARN`).
#[test]
fn a_chunk_the_executor_caps_reports_the_call_size_in_effect() {
    let capped = PrefillChunkOutcome {
        kind: PrefillChunkKind::Hybrid,
        requested: 4096,
        honoured: 4096,
        in_effect: 1536,
    };
    assert!(capped.capped_by_executor());
    let line = capped.message();
    assert!(
        line.starts_with("hybrid Gated-DeltaNet prefill chunk in effect is 1536 tokens"),
        "{line}"
    );
    assert!(line.contains("--prefill-chunk asked for 4096"), "{line}");
    assert!(line.contains("KV window"), "{line}");
    assert!(!line.contains("set to"), "{line}");

    // A smaller request is taken as given by every executor.
    let smaller = PrefillChunkOutcome {
        kind: PrefillChunkKind::Hybrid,
        requested: 64,
        honoured: 64,
        in_effect: 64,
    };
    assert!(!smaller.capped_by_executor());
}

/// On every executor this host builds for the synthetic hybrid — the CPU
/// model, and the Metal hybrid runner where there is one — the outcome
/// reports the chunk the ENGINE says is in effect
/// (`InferenceEngine::prefill_chunk_in_effect`), read after the flag was
/// applied at build time (`EngineLoad::prefill_chunk` reaches the runner
/// through the hybrid load options): the runner's calls are sized for the
/// request, so a chunk larger than the model's default 512 is in effect, not
/// only "set". The expectation is the engine's, never "Metal present".
///
/// The build-time plumbing is pinned on the runner itself: straight after
/// the engine is built, before any prefill (which fits the runner's calls to
/// the model's chunk whatever the engine was built with), the runner's live
/// call size is the requested chunk, bounded by the footprint cap the
/// engine's planner applies (see [`assert_runner_built_for_chunk`]).
#[test]
fn the_logged_chunk_is_the_one_the_engines_executor_runs_in() {
    let bytes = oxibonsai_testkit::qwen35_fixture::synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("the fixture parses");
    for backend in [
        oxibonsai_runtime::engine_seam::Backend::Cpu,
        oxibonsai_runtime::engine_seam::Backend::Auto,
        oxibonsai_runtime::engine_seam::Backend::Metal,
    ] {
        for requested in [5usize, 1000] {
            let load = crate::cli::cmd_run::EngineLoad {
                prefill_chunk: Some(requested),
                ..prefill_chunk_load(backend)
            };
            let Ok(mut engine) = crate::cli::cmd_run::load_engine(&gguf, &load, 0) else {
                assert_eq!(
                    backend,
                    oxibonsai_runtime::engine_seam::Backend::Metal,
                    "only an explicit Metal backend may be unavailable on this host"
                );
                continue;
            };
            assert_runner_built_for_chunk(&engine, backend, requested);
            let in_effect = engine.prefill_chunk_in_effect();
            let outcome = apply_prefill_chunk(&mut engine, Some(requested))
                .expect("a valid chunk")
                .expect("a flag was given");
            assert_eq!(outcome.requested, requested, "{backend}");
            assert_eq!(
                outcome.in_effect,
                engine.prefill_chunk_in_effect(),
                "{backend}: the logged chunk is the engine's own"
            );
            assert_eq!(
                outcome.in_effect, in_effect,
                "{backend}: applying the flag again changes nothing the build did not"
            );
            assert_eq!(
                outcome.honoured,
                engine.hybrid_model().expect("hybrid").prefill_chunk(),
                "{backend}"
            );
            let line = outcome.message();
            if outcome.capped_by_executor() {
                assert!(
                    line.contains(&format!("in effect is {} tokens", outcome.in_effect)),
                    "{backend}: {line}"
                );
                assert!(
                    line.contains(&format!("asked for {requested}")),
                    "{backend}: {line}"
                );
            } else {
                assert_eq!(outcome.in_effect, requested, "{backend}");
                assert_eq!(
                    line,
                    format!("hybrid Gated-DeltaNet prefill chunk set to {requested} tokens"),
                    "{backend}"
                );
            }
            // The engine's prefill really runs in calls of that size: a
            // prompt longer than one call prefills without error.
            let prompt: Vec<u32> = (0..(outcome.in_effect + 3).min(40) as u32).collect();
            let logits = engine.prefill_from_pos(&prompt, 0).expect("prefill");
            assert!(logits.iter().all(|v| v.is_finite()), "{backend}");
        }
    }
}

/// A Metal-backed engine's runner is built for the requested chunk: its live
/// call size (`InferenceEngine::hybrid_metal_call_tokens`), read before any
/// prefill, is the request — or, when the planner's footprint cap binds, the
/// cap the engine reports in effect (`prefill_chunk_in_effect`, the call size
/// its next prefill takes). A runner built outside the `HybridLoadScope` that
/// `cmd_run::construct_engine` installs keeps the model's default call size
/// and only fits it to the chunk at its first prefill, so it differs from
/// both.
///
/// Whether the engine has a runner is the engine's own answer
/// (`hybrid_backend`), never "Metal present": an engine built with
/// `--backend metal` always has one, `--backend cpu` never, and `auto` is
/// whatever it resolved to.
fn assert_runner_built_for_chunk(
    engine: &oxibonsai_runtime::InferenceEngine<'_>,
    backend: oxibonsai_runtime::engine_seam::Backend,
    requested: usize,
) {
    use oxibonsai_runtime::engine_hybrid_gpu::HybridBackend;
    use oxibonsai_runtime::engine_seam::Backend;

    let executor = engine.hybrid_backend();
    match backend {
        Backend::Metal => assert_eq!(
            executor,
            Some(HybridBackend::Metal),
            "{backend}: an explicit Metal engine decodes on the runner"
        ),
        Backend::Cpu => assert_eq!(
            executor,
            Some(HybridBackend::Cpu),
            "{backend}: a CPU engine decodes on the CPU model"
        ),
        Backend::Auto => {}
    }
    let built = engine.hybrid_metal_call_tokens();
    assert_eq!(
        built.is_some(),
        executor == Some(HybridBackend::Metal),
        "{backend}: the engine has a runner call size exactly when it decodes on the runner"
    );
    let Some(built) = built else {
        return;
    };
    assert_ne!(
        requested,
        oxibonsai_model::hybrid::DEFAULT_PREFILL_CHUNK,
        "{backend}: only a request that is not the model's default call size tells a runner \
         built for it from one built without"
    );
    // The request, bounded by the footprint cap the engine's planner applies.
    let expected = requested.min(engine.prefill_chunk_in_effect());
    assert_eq!(
        built,
        expected,
        "{backend}: the runner was built to take calls of {built} tokens, not the {expected} \
         requested through the hybrid load options (a runner built outside `HybridLoadScope` \
         keeps the model's default {} until its first prefill)",
        oxibonsai_model::hybrid::DEFAULT_PREFILL_CHUNK
    );
}

fn prefill_chunk_load(
    backend: oxibonsai_runtime::engine_seam::Backend,
) -> crate::cli::cmd_run::EngineLoad {
    crate::cli::cmd_run::EngineLoad {
        params: oxibonsai_runtime::sampling::SamplingParams {
            temperature: 0.0,
            ..oxibonsai_runtime::sampling::SamplingParams::default()
        },
        seed: 7,
        max_seq_len: 64,
        backend,
        rope_scaling: oxibonsai_runtime::config::RopeScalingMode::Auto,
        prefill_chunk: None,
        penalties: oxibonsai_runtime::sampling::PenaltyParams::default(),
        min_p: 0.0,
        wants_vision: false,
        vision_resident_bytes: 0,
    }
}

/// On a real (synthetic-fixture) hybrid engine: the outcome carries the
/// chunk read back from the model, the model holds it, and no flag is no
/// outcome.
#[test]
fn apply_prefill_chunk_reports_the_chunk_the_hybrid_model_holds() {
    let bytes = oxibonsai_testkit::qwen35_fixture::synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("the fixture parses");
    let mut engine = crate::cli::cmd_run::load_engine(
        &gguf,
        &prefill_chunk_load(oxibonsai_runtime::engine_seam::Backend::Cpu),
        0,
    )
    .expect("a CPU hybrid engine");
    let default_chunk = engine.hybrid_model().expect("hybrid").prefill_chunk();
    assert_ne!(default_chunk, 5, "the request below changes something");

    assert_eq!(
        apply_prefill_chunk(&mut engine, None).expect("no flag"),
        None
    );
    assert_eq!(
        engine.hybrid_model().expect("hybrid").prefill_chunk(),
        default_chunk,
        "no flag leaves the model's own default"
    );

    let outcome = apply_prefill_chunk(&mut engine, Some(5))
        .expect("a valid chunk")
        .expect("a flag was given");
    assert_eq!(outcome.kind, PrefillChunkKind::Hybrid);
    assert_eq!(outcome.requested, 5);
    assert_eq!(
        outcome.honoured,
        engine.hybrid_model().expect("hybrid").prefill_chunk(),
        "the reported chunk is the one the model holds"
    );
    assert_eq!(
        outcome.message(),
        "hybrid Gated-DeltaNet prefill chunk set to 5 tokens"
    );

    let refused = apply_prefill_chunk(&mut engine, Some(0)).expect_err("a zero chunk");
    assert!(
        refused.to_string().contains("--prefill-chunk 0"),
        "{refused}"
    );
}

/// The same on a dense engine, whose chunk is the chunked-prefill size.
#[test]
fn apply_prefill_chunk_reports_the_chunk_the_dense_model_holds() {
    let bytes = crate::cli::test_fixtures::tiny_dense_gguf(Vec::new());
    let gguf = GgufFile::parse(&bytes).expect("parse");
    let mut engine = crate::cli::cmd_run::load_engine(
        &gguf,
        &prefill_chunk_load(oxibonsai_runtime::engine_seam::Backend::Cpu),
        0,
    )
    .expect("a dense engine");

    let outcome = apply_prefill_chunk(&mut engine, Some(8))
        .expect("a valid chunk")
        .expect("a flag was given");
    assert_eq!(outcome.kind, PrefillChunkKind::Dense);
    assert_eq!((outcome.requested, outcome.honoured), (8, 8));
    assert_eq!(
        engine.dense_model().expect("dense").prefill_chunk_tokens(),
        outcome.honoured
    );
    assert_eq!(
        outcome.message(),
        "dense chunked-prefill size set to 8 tokens"
    );

    let refused = apply_prefill_chunk(&mut engine, Some(0)).expect_err("a zero chunk");
    assert!(refused.to_string().contains(">= 1"), "{refused}");
}

/// `EngineLoad::prefill_chunk` is applied while the engine is built, through
/// the same function.
#[test]
fn the_engine_load_applies_the_prefill_chunk_flag() {
    let bytes = oxibonsai_testkit::qwen35_fixture::synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("the fixture parses");
    let load = crate::cli::cmd_run::EngineLoad {
        prefill_chunk: Some(3),
        ..prefill_chunk_load(oxibonsai_runtime::engine_seam::Backend::Cpu)
    };
    let engine = crate::cli::cmd_run::load_engine(&gguf, &load, 0).expect("a CPU hybrid engine");
    assert_eq!(engine.hybrid_model().expect("hybrid").prefill_chunk(), 3);
}
