//! Unit tests for `vision_prefill.rs` (declared there via `#[path]`, so
//! `super` still names that module).

use super::*;
use crate::engine_seam::{engine_error_code, Backend};
use crate::sampling::SamplingParams;
use oxibonsai_testkit::mmproj_fixture::{pattern_rgb8, synthetic_mmproj_gguf, MmprojFixtureSpec};
use oxibonsai_testkit::qwen35_fixture::{synthetic_qwen35_gguf, HIDDEN, VOCAB};

const MAX_SEQ: usize = 64;
const IDS: VisionTokenIds = VisionTokenIds {
    vision_start: 500,
    vision_end: 501,
    image_pad: 502,
};

fn greedy() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        ..SamplingParams::default()
    }
}

/// A hybrid engine decoding on the CPU model, chosen explicitly rather
/// than left to `Backend::Auto`.
fn cpu_engine<'a>(gguf: &'a GgufFile<'a>) -> InferenceEngine<'a> {
    InferenceEngine::from_gguf_with_backend(gguf, greedy(), 7, MAX_SEQ, Backend::Cpu)
        .expect("a CPU hybrid engine")
}

/// Whether this build and host give a hybrid engine the Metal runner. A
/// missing device is the only "no"; any other failure to open the
/// device on a Metal build is a test failure, never a skip.
fn metal_available() -> bool {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        match oxibonsai_kernels::MetalGraph::shared_device() {
            Ok(_) => true,
            Err(oxibonsai_kernels::MetalGraphError::DeviceNotFound) => false,
            Err(e) => panic!("the Metal device must open on this host: {e}"),
        }
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        false
    }
}

/// A hybrid engine decoding on the Metal runner.
fn metal_engine<'a>(gguf: &'a GgufFile<'a>) -> InferenceEngine<'a> {
    let engine =
        InferenceEngine::from_gguf_with_backend(gguf, greedy(), 7, MAX_SEQ, Backend::Metal)
            .expect("a Metal hybrid engine");
    assert_eq!(engine.hybrid_backend(), Some(HybridBackend::Metal));
    engine
}

/// The engines of every executor this host has, CPU first.
fn engines<'a>(gguf: &'a GgufFile<'a>) -> Vec<InferenceEngine<'a>> {
    let mut engines = vec![cpu_engine(gguf)];
    if metal_available() {
        engines.push(metal_engine(gguf));
    }
    engines
}

/// The test kit's tiny projector, widened to the fixture model's hidden
/// size, written under the temp dir (`tag` keeps parallel tests apart).
fn projector_file(tag: &str) -> std::path::PathBuf {
    projector_file_of_width(tag, HIDDEN)
}

/// [`projector_file`] emitting `width`-wide rows.
fn projector_file_of_width(tag: &str, width: usize) -> std::path::PathBuf {
    let spec = MmprojFixtureSpec {
        projection_dim: width,
        ..MmprojFixtureSpec::tiny()
    };
    let bytes = synthetic_mmproj_gguf(&spec).expect("synthetic projector");
    let path = std::env::temp_dir().join(format!(
        "oxibonsai_vision_prefill_{tag}_{}.gguf",
        std::process::id()
    ));
    std::fs::write(&path, bytes).expect("write the projector");
    path
}

fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |best, (i, &v)| {
            if v > best.1 {
                (i, v)
            } else {
                best
            }
        })
        .0
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += f64::from(x) * f64::from(y);
        na += f64::from(x) * f64::from(x);
        nb += f64::from(y) * f64::from(y);
    }
    dot / (na.sqrt() * nb.sqrt()).max(f64::MIN_POSITIVE)
}

fn image(grid: GridSize, seed: u32) -> EncodedImage {
    let rows = (0..grid.n_tokens() * HIDDEN)
        .map(|i| {
            let x = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
            ((x >> 8) as f32 / 16_777_216.0) - 0.5
        })
        .collect();
    EncodedImage {
        rows,
        grid,
        source: (grid.w * 32, grid.h * 32),
    }
}

fn prompt(grid: GridSize) -> MultimodalPrompt {
    let tokens = vec![
        10,
        11,
        IDS.vision_start,
        IDS.image_pad,
        IDS.vision_end,
        12,
        13,
    ];
    MultimodalPrompt::new(tokens, vec![image(grid, 9)], IDS).expect("splices")
}

/// A Metal-backed engine serves multimodal prefill on its runner (it
/// used to refuse): the same rows give the CPU engine's greedy ids at
/// every step, the same M-RoPE offset and sequence position, and a
/// prefill-logit row within float rounding of the CPU one.
#[test]
fn a_metal_engine_serves_multimodal_prefill_like_the_cpu_engine() {
    if !metal_available() {
        return;
    }
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let mut cpu = cpu_engine(&gguf);
    let mut metal = metal_engine(&gguf);
    for grid in [GridSize { h: 2, w: 2 }, GridSize { h: 3, w: 5 }] {
        let p = prompt(grid);
        let want = cpu
            .prefill_multimodal(&p.segments(), 0)
            .expect("cpu prefill");
        let got = metal
            .prefill_multimodal(&p.segments(), 0)
            .expect("metal prefill");
        let cos = cosine(&got, &want);
        assert!(cos >= 0.9999, "{grid:?}: prefill logit cosine {cos}");
        assert_eq!(argmax(&got), argmax(&want), "{grid:?}");
        assert_eq!(metal.sequence_position(), p.len());
        assert_eq!(metal.rope_delta(), grid.n_tokens() - grid.h.max(grid.w));
        assert_eq!(metal.rope_delta(), cpu.rope_delta());
        // The CPU model beside the runner never saw the rows.
        assert_eq!(metal.hybrid_model().map(|m| m.rope_delta()), Some(0));

        let want_ids = cpu.generate_multimodal(&p, 8).expect("cpu generate");
        let got_ids = metal.generate_multimodal(&p, 8).expect("metal generate");
        assert_eq!(got_ids, want_ids, "{grid:?}: greedy ids");
        assert!(!got_ids.is_empty());
    }
}

/// Every multimodal prefill counts its rows (images expanded) into the
/// engine's prefill statistics, exactly as a text prefill counts its
/// tokens, on either executor.
#[test]
fn multimodal_prefill_counts_every_row_on_either_executor() {
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    for mut engine in engines(&gguf) {
        let backend = engine.hybrid_backend();
        assert_eq!(engine.prefill_token_count(), 0);
        let p = prompt(GridSize { h: 2, w: 3 });
        engine
            .prefill_multimodal(&p.segments(), 0)
            .expect("prefill");
        assert_eq!(engine.prefill_token_count(), p.len() as u64, "{backend:?}");
        engine
            .prefill_from_pos(&[40, 41, 42], p.len())
            .expect("text continuation");
        assert_eq!(
            engine.prefill_token_count(),
            p.len() as u64 + 3,
            "{backend:?}"
        );
        // A refused prefill records nothing.
        engine
            .prefill_multimodal(&p.segments(), p.len() + 9)
            .expect_err("gap");
        assert_eq!(
            engine.prefill_token_count(),
            p.len() as u64 + 3,
            "{backend:?}"
        );
        engine.record_prefill_tokens(5);
        assert_eq!(
            engine.prefill_token_count(),
            p.len() as u64 + 8,
            "{backend:?}"
        );
    }
}

/// A Metal engine's vision service holds only the Metal tower (the CPU
/// tower's `f32` weights are never built), whose footprint is known
/// before it is built and whose rows track the CPU tower's; and the
/// whole pipeline — Metal tower, Metal runner — generates the CPU
/// pipeline's greedy ids.
#[test]
fn a_metal_engine_encodes_on_the_metal_tower_and_generates_the_cpu_ids() {
    let path = projector_file("pipeline");
    let policy = ImageSourcePolicy::default();
    let cpu_service = VisionService::load(&path, 64, policy.clone()).expect("cpu tower");
    assert_eq!(cpu_service.tower().backend_name(), "cpu");
    assert!(!cpu_service.tower().is_metal());
    let same = VisionService::load_for_backend(&path, 64, policy.clone(), Some(HybridBackend::Cpu))
        .expect("cpu tower by backend");
    assert_eq!(same.tower().backend_name(), "cpu");
    if !metal_available() {
        let _ = std::fs::remove_file(&path);
        return;
    }
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let mut cpu = cpu_engine(&gguf);
    let mut metal = metal_engine(&gguf);
    let metal_service = VisionService::load_for_backend(&path, 64, policy, metal.hybrid_backend())
        .expect("metal tower");
    assert!(metal_service.tower().is_metal());
    assert_eq!(metal_service.preprocess(), cpu_service.preprocess());
    let footprint = VisionService::metal_footprint(&path, 64).expect("footprint");
    assert_eq!(footprint, metal_service.tower().resident_bytes() as u64);
    let _ = std::fs::remove_file(&path);

    let image = oxibonsai_model::vision::ImageRgb8::new(96, 64, pattern_rgb8(96, 64))
        .expect("pattern image");
    let want = cpu_service.encode_image(0, &image).expect("cpu encode");
    let got = metal_service.encode_image(0, &image).expect("metal encode");
    assert_eq!(got.grid, want.grid);
    assert!(got.grid.n_tokens() >= 8, "{:?}", got.grid);
    for (row, (g, w)) in got
        .rows
        .chunks(HIDDEN)
        .zip(want.rows.chunks(HIDDEN))
        .enumerate()
    {
        let cos = cosine(g, w);
        assert!(cos >= 0.999, "image row {row}: cosine {cos}");
    }
    let tokens = vec![
        10,
        11,
        IDS.vision_start,
        IDS.image_pad,
        IDS.vision_end,
        12,
        13,
    ];
    let cpu_prompt = MultimodalPrompt::new(tokens.clone(), vec![want], IDS).expect("splices");
    let metal_prompt = MultimodalPrompt::new(tokens, vec![got], IDS).expect("splices");
    let want_ids = cpu.generate_multimodal(&cpu_prompt, 8).expect("cpu");
    let got_ids = metal.generate_multimodal(&metal_prompt, 8).expect("metal");
    assert_eq!(got_ids, want_ids, "greedy ids, tower and runner on Metal");
}

/// Every constructor bounds the source decode by the token budget: a
/// library caller that hands `VisionService::new` (or `load` / the engine
/// loaders, which all go through `with_encoder`) a policy without a decode
/// budget gets the one the preprocessing derives, so a small file declaring
/// a huge image is refused from its header; a policy that already carries a
/// tighter budget keeps it, and a looser one is tightened to the derived one.
#[test]
fn every_constructor_bounds_the_source_decode_by_the_token_budget() {
    let spec = MmprojFixtureSpec {
        projection_dim: HIDDEN,
        ..MmprojFixtureSpec::tiny()
    };
    let bytes = synthetic_mmproj_gguf(&spec).expect("synthetic projector");
    let gguf = GgufFile::parse(&bytes).expect("parse");
    let tower = || VisionTower::from_mmproj(&gguf).expect("cpu tower");
    for budget in [16usize, 64, 1024] {
        let service =
            VisionService::new(tower(), budget, ImageSourcePolicy::default()).expect("a budget");
        let derived = service.preprocess().max_source_pixels();
        assert_eq!(
            service.policy().max_source_pixels,
            Some(derived),
            "budget {budget}: the derived decode bound"
        );
        assert_eq!(
            ImageSourcePolicy::default()
                .with_token_budget(budget)
                .max_source_pixels,
            Some(derived),
            "budget {budget}: the same bound the CLI and the server derive"
        );

        let tighter = ImageSourcePolicy {
            max_source_pixels: Some(derived / 2),
            ..ImageSourcePolicy::default()
        };
        let kept = VisionService::new(tower(), budget, tighter).expect("a budget");
        assert_eq!(kept.policy().max_source_pixels, Some(derived / 2));

        let looser = ImageSourcePolicy {
            max_source_pixels: Some(derived.saturating_mul(4)),
            ..ImageSourcePolicy::default()
        };
        let tightened = VisionService::new(tower(), budget, looser).expect("a budget");
        assert_eq!(tightened.policy().max_source_pixels, Some(derived));
    }
    // The loaders go through the same constructor.
    let path = projector_file("decode_budget");
    let loaded =
        VisionService::load(&path, 64, ImageSourcePolicy::default()).expect("the projector loads");
    let _ = std::fs::remove_file(&path);
    assert_eq!(
        loaded.policy().max_source_pixels,
        Some(loaded.preprocess().max_source_pixels())
    );
}

/// The header-only footprints are what the built towers keep resident.
#[test]
fn the_footprints_are_what_the_towers_keep() {
    let spec = MmprojFixtureSpec {
        projection_dim: HIDDEN,
        ..MmprojFixtureSpec::tiny()
    };
    let bytes = synthetic_mmproj_gguf(&spec).expect("synthetic projector");
    let gguf = GgufFile::parse(&bytes).expect("parse");
    let fp = VisionService::footprints(&gguf, 64).expect("footprints");
    assert_eq!(fp.blocks, spec.blocks);
    assert_eq!(fp.image_max_tokens, 64);
    let cpu = VisionTower::from_mmproj(&gguf).expect("cpu tower");
    assert_eq!(fp.cpu_bytes, cpu.resident_bytes() as u64);
    let path = projector_file("footprints");
    let metal_bytes = VisionService::metal_footprint(&path, 64).expect("metal footprint");
    let _ = std::fs::remove_file(&path);
    assert_eq!(fp.metal_bytes.unwrap_or(0), metal_bytes);
    if cfg!(all(feature = "metal", target_os = "macos")) {
        assert!(fp.metal_bytes.is_some_and(|b| b > 0));
    }
    assert!(VisionService::footprints(&gguf, 0).is_err() || fp.metal_bytes.is_none());
}

/// The service is shared across request threads.
#[test]
fn the_vision_service_is_send_and_sync() {
    fn shared<T: Send + Sync>() {}
    shared::<VisionService>();
    shared::<Arc<VisionService>>();
}

/// The seam's refusal shape, for an executor without a rows prefill: an
/// engine error with a stable code whose message names the executor it is
/// handed and the constraint. No hybrid executor earns it any more (the
/// Metal runner has its own rows prefill), so the constraint it names is
/// the split sequence, no longer "CPU model only".
#[test]
fn the_metal_backend_refusal_is_a_typed_engine_error_naming_the_constraint() {
    let err = multimodal_backend_refusal(Backend::Metal, "qwen35");
    assert_eq!(engine_error_code(&err), Some("BACKEND_UNAVAILABLE"));
    let text = err.to_string();
    assert!(text.contains("vision (multimodal) prefill"), "{text}");
    assert!(text.contains("the metal executor"), "{text}");
    assert!(text.contains("never see the image rows"), "{text}");
    assert!(text.contains("`qwen35`"), "{text}");
}

/// A CPU hybrid engine prefills image rows on the CPU model: the pre-encode
/// check names that executor and the predicate built on it agrees.
#[test]
fn a_cpu_hybrid_engine_prefills_images_on_the_cpu_model() {
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let engine = cpu_engine(&gguf);
    assert_eq!(
        engine.multimodal_executor().expect("served"),
        HybridBackend::Cpu
    );
    assert!(engine.prefills_images());
}

/// Both hybrid executors serve image turns: the pre-encode check names the
/// executor the engine decodes on — what `auto` resolved to, never the
/// backend it was asked for — and `prefills_images` answers `true` on each.
#[test]
fn every_hybrid_executor_names_itself_before_any_encode() {
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    for engine in engines(&gguf) {
        let executor = engine.hybrid_backend().expect("a hybrid engine");
        assert_eq!(engine.multimodal_executor().expect("served"), executor);
        assert!(engine.prefills_images(), "{executor}");
        assert_eq!(engine.prefill_token_count(), 0, "nothing ran");
    }
    let auto = InferenceEngine::from_gguf_with_backend(&gguf, greedy(), 7, MAX_SEQ, Backend::Auto)
        .expect("an auto hybrid engine");
    assert_eq!(auto.backend(), Backend::Auto);
    assert_eq!(auto.multimodal_executor().ok(), auto.hybrid_backend());
    assert_eq!(executor_label(HybridBackend::Cpu), "CPU hybrid model");
    assert_eq!(executor_label(HybridBackend::Metal), "Metal hybrid runner");
}

/// A projector and an engine are checked together when the projector is
/// loaded for the engine: a dense engine is refused before the file is
/// even opened, a projector whose rows are not as wide as the model's
/// residual stream is refused (typed, naming both widths and the
/// executor), and a matching one gets the tower of the engine's own
/// executor.
#[test]
fn the_projector_and_the_engine_are_checked_together_at_load() {
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let fits = projector_file("load_for_engine");
    let narrow = projector_file_of_width("load_for_engine_narrow", HIDDEN / 2);
    let policy = ImageSourcePolicy::default();
    for engine in engines(&gguf) {
        let executor = engine.hybrid_backend().expect("a hybrid engine");
        let service = VisionService::load_for_engine(&fits, 64, policy.clone(), &engine)
            .expect("a matching projector loads");
        assert_eq!(
            service.tower().is_metal(),
            executor == HybridBackend::Metal,
            "{executor}"
        );
        assert_eq!(service.check_engine(&engine).expect("fits"), executor);

        let err = VisionService::load_for_engine(&narrow, 64, policy.clone(), &engine)
            .expect_err("rows of another width");
        let text = err.to_string();
        assert!(text.contains("[projector_mismatch]"), "{text}");
        assert!(
            text.contains(&format!("emits {}-wide image rows", HIDDEN / 2)),
            "{text}"
        );
        assert!(
            text.contains(&format!("reads {HIDDEN}-wide rows")),
            "{text}"
        );
        assert!(text.contains(executor_label(executor)), "{text}");
        assert_eq!(engine.prefill_token_count(), 0, "nothing was prefilled");
    }
    let dense = InferenceEngine::new(
        oxibonsai_core::config::Qwen3Config::tiny_test(),
        greedy(),
        7,
    );
    let missing = std::env::temp_dir().join(format!(
        "oxibonsai_vision_prefill_no_such_projector_{}.gguf",
        std::process::id()
    ));
    let err =
        VisionService::load_for_engine(&missing, 64, policy, &dense).expect_err("a dense engine");
    assert_eq!(engine_error_code(&err), Some("NOT_A_HYBRID_MODEL"), "{err}");
    let _ = std::fs::remove_file(&fits);
    let _ = std::fs::remove_file(&narrow);
}

#[test]
fn multimodal_prompts_expand_each_placeholder_to_its_grid() {
    let grid = GridSize { h: 2, w: 3 };
    let p = prompt(grid);
    assert_eq!(p.len(), 7 - 1 + 6);
    assert_eq!(p.images().len(), 1);
    let segments = p.segments();
    assert_eq!(segments.len(), 3);
    assert_eq!(
        segments[0],
        PromptSegment::Text(vec![10, 11, IDS.vision_start])
    );
    assert!(matches!(&segments[1], PromptSegment::Image { grid: g, .. } if *g == grid));
    assert_eq!(
        segments[2],
        PromptSegment::Text(vec![IDS.vision_end, 12, 13])
    );
    let chat = ChatPrompt::Multimodal(p);
    assert_eq!(chat.len(), 12);
    assert_eq!(chat.image_count(), 1);
    assert_eq!(chat.tokens().len(), 7);
    assert_eq!(ChatPrompt::Text(vec![1, 2]).len(), 2);

    let err = MultimodalPrompt::new(vec![1, IDS.image_pad], vec![image(grid, 1)], IDS)
        .expect_err("unbracketed");
    assert_eq!(err.code(), "image_placeholder_misplaced");
    let err = MultimodalPrompt::new(vec![1, 2], vec![image(grid, 1)], IDS).expect_err("none");
    assert_eq!(err.code(), "image_placeholder_count_mismatch");
    assert!(err.is_client_error());
}

#[test]
fn a_dense_engine_refuses_multimodal_prefill_by_name() {
    let mut engine = InferenceEngine::new(
        oxibonsai_core::config::Qwen3Config::tiny_test(),
        greedy(),
        7,
    );
    // The pre-encode check refuses it the same way, reading no state:
    // the refusal is for the model kind, not for the executor, and
    // `prefills_images` answers `false`.
    let early = engine.multimodal_executor().expect_err("dense");
    assert_eq!(engine_error_code(&early), Some("NOT_A_HYBRID_MODEL"));
    assert!(!engine.prefills_images());
    let err = engine
        .prefill_multimodal(&prompt(GridSize { h: 1, w: 1 }).segments(), 0)
        .expect_err("dense");
    assert_eq!(err.to_string(), early.to_string());
    assert!(err.to_string().contains("[NOT_A_HYBRID_MODEL]"), "{err}");
    // The engine layer: the typed refusal, naming the operation and the
    // dense architecture.
    assert_eq!(engine_error_code(&err), Some("NOT_A_HYBRID_MODEL"));
    let RuntimeError::Engine(EngineError::NotAHybridModel {
        operation,
        architecture,
    }) = &err
    else {
        panic!("expected the typed NotAHybridModel refusal, got {err:?}");
    };
    assert_eq!(*operation, MULTIMODAL_PREFILL_OPERATION);
    assert_eq!(architecture, engine.architecture());
    assert_eq!(
        EngineError::NotAHybridModel {
            operation: MULTIMODAL_PREFILL_OPERATION,
            architecture: "qwen3".into(),
        }
        .to_string(),
        "multimodal (image) prefill requires a hybrid `qwen35` (Bonsai 2) model, but this \
         engine holds a dense `qwen3` model"
    );
    // The request-level error converts to the same typed refusal.
    let converted = RuntimeError::from(MultimodalError::NotAHybridModel {
        architecture: "qwen3".into(),
    });
    assert_eq!(engine_error_code(&converted), Some("NOT_A_HYBRID_MODEL"));
    assert_eq!(engine.prefill_token_count(), 0);
}

/// The HTTP layer keeps its mapping: a multimodal request a dense
/// server cannot serve is a `400` whose code is `NOT_A_HYBRID_MODEL`.
#[cfg(feature = "server")]
#[test]
fn the_http_mapping_of_a_dense_refusal_is_unchanged() {
    let api = crate::tokenizer_bridge::chat_render::api_error_from_multimodal(
        &MultimodalError::NotAHybridModel {
            architecture: "qwen3".into(),
        },
    );
    assert_eq!(api.status(), axum::http::StatusCode::BAD_REQUEST);
    let json = api.to_json();
    assert_eq!(json["error"]["code"], "NOT_A_HYBRID_MODEL", "{json}");
    let message = json["error"]["message"].as_str().unwrap_or_default();
    assert!(message.contains("dense `qwen3` model"), "{json}");
}

/// The engine's multimodal prefill is the model's rows prefill (same
/// logits, bit for bit), advances the sequence by the expanded length,
/// and every generation form agrees with the plain one.
#[test]
fn hybrid_engine_prefills_and_generates_from_a_multimodal_prompt() {
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let grid = GridSize { h: 2, w: 2 };
    let p = prompt(grid);

    let mut engine = cpu_engine(&gguf);
    let logits = engine
        .prefill_multimodal(&p.segments(), 0)
        .expect("prefill");
    assert_eq!(logits.len(), VOCAB);
    assert_eq!(engine.sequence_position(), p.len());
    let delta = engine.hybrid_model().map(|m| m.rope_delta());
    assert_eq!(delta, Some(grid.n_tokens() - 2));
    assert_eq!(engine.hybrid_backend(), Some(HybridBackend::Cpu));
    assert_eq!(Some(engine.rope_delta()), delta);

    // The model-level rows prefill of the same pieces.
    let mut direct =
        oxibonsai_model::hybrid::HybridModel::from_gguf(&gguf, MAX_SEQ).expect("model");
    let segments = p.segments();
    let pieces: Vec<PromptPiece<'_>> = segments
        .iter()
        .map(|s| match s {
            PromptSegment::Text(t) => PromptPiece::Tokens(t),
            PromptSegment::Image { rows, grid } => PromptPiece::Image { rows, grid: *grid },
        })
        .collect();
    let mut want = vec![0.0f32; VOCAB];
    direct
        .forward_prefill_pieces(&pieces, 0, Some(&mut want))
        .expect("direct prefill");
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&logits), bits(&want));

    // A continuation at a non-contiguous position is refused.
    let err = engine
        .prefill_multimodal(&p.segments(), p.len() + 3)
        .expect_err("gap");
    assert_eq!(engine_error_code(&err), Some("NON_CONTIGUOUS_POSITION"));

    // Generation: plain, logprobs and both streaming forms agree.
    let plain = engine.generate_multimodal(&p, 6).expect("generate");
    let (with_lp, logprobs) = engine
        .generate_multimodal_with_logprobs(&p, 6, 3, &|id| format!("<{id}>"))
        .expect("logprobs");
    assert_eq!(with_lp, plain);
    assert_eq!(logprobs.len(), plain.len());
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let n = engine
        .generate_multimodal_streaming(&p, 6, &tx)
        .expect("stream");
    drop(tx);
    let mut streamed = Vec::new();
    while let Ok(t) = rx.try_recv() {
        streamed.push(t);
    }
    assert_eq!(n, plain.len());
    assert_eq!(streamed, plain);
    let (stx, srx) = std::sync::mpsc::channel();
    let n = engine
        .generate_multimodal_streaming_sync(&p, 6, &stx)
        .expect("stream sync");
    drop(stx);
    assert_eq!(srx.iter().collect::<Vec<_>>(), plain);
    assert_eq!(n, plain.len());
    // Through the prompt enum.
    let chat = ChatPrompt::Multimodal(p.clone());
    assert_eq!(chat.generate(&mut engine, 6).expect("enum"), plain);

    // The first generated token is the argmax of the prefill's logits.
    if let Some(&first) = plain.first() {
        let best = logits
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |b, (i, &v)| {
                if v > b.1 {
                    (i, v)
                } else {
                    b
                }
            });
        assert_eq!(first as usize, best.0);
    }
}

/// A sequence snapshotted at `k`, continued by
/// `prefill_multimodal(segments, k)` and then restored continues exactly
/// like an engine that never saw the image: the model's M-RoPE offset
/// follows the restored position (the tail and the decode step after it
/// run past where the abandoned continuation ended).
#[test]
fn a_restored_sequence_continues_at_its_own_rope_offset() {
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let mut engine = cpu_engine(&gguf);
    let mut reference = cpu_engine(&gguf);
    let lead = [10u32, 11, 12, 13];
    let tail = [14u32, 15, 16, 17, 18, 19, 20, 21, 22];
    let k = lead.len();
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();

    engine.prefill_from_pos(&lead, 0).expect("prefill");
    let snapshot = engine.snapshot_sequence().expect("snapshot");
    let grid = GridSize { h: 2, w: 3 };
    let continuation = [
        PromptSegment::Image {
            rows: image(grid, 21).rows,
            grid,
        },
        PromptSegment::Text(vec![30]),
    ];
    engine
        .prefill_multimodal(&continuation, k)
        .expect("image continuation");
    let delta = |e: &InferenceEngine<'_>| e.hybrid_model().map(|m| m.rope_delta());
    assert_eq!(delta(&engine), Some(grid.n_tokens() - 3));
    engine.restore_sequence(&snapshot).expect("restore");
    assert_eq!(engine.sequence_position(), k);
    assert_eq!(delta(&engine), Some(0), "the offset in force at k");
    let got = engine.prefill_from_pos(&tail, k).expect("continue");
    let got_next = engine.decode_step(23, k + tail.len()).expect("decode");

    reference.prefill_from_pos(&lead, 0).expect("prefill");
    let want = reference.prefill_from_pos(&tail, k).expect("continue");
    let want_next = reference.decode_step(23, k + tail.len()).expect("decode");
    assert_eq!(bits(&got), bits(&want));
    assert_eq!(bits(&got_next), bits(&want_next));
}

/// The Metal runner's snapshot carries its M-RoPE offsets: a sequence
/// snapshotted at `k`, continued by `prefill_multimodal(segments, k)`
/// and restored continues exactly like one that never saw the image,
/// and a snapshot taken after an image restores that image's offset.
#[test]
fn a_metal_snapshot_restores_the_rope_offset_across_a_multimodal_continuation() {
    if !metal_available() {
        return;
    }
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let mut engine = metal_engine(&gguf);
    let mut reference = metal_engine(&gguf);
    let lead = [10u32, 11, 12, 13];
    let tail = [14u32, 15, 16, 17, 18, 19, 20, 21, 22];
    let k = lead.len();
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let grid = GridSize { h: 2, w: 3 };
    let continuation = [
        PromptSegment::Image {
            rows: image(grid, 21).rows,
            grid,
        },
        PromptSegment::Text(vec![30]),
    ];

    engine.prefill_from_pos(&lead, 0).expect("prefill");
    let before = engine.snapshot_sequence().expect("snapshot");
    assert_eq!(before.rope_delta(), 0);
    engine
        .prefill_multimodal(&continuation, k)
        .expect("image continuation");
    let delta = grid.n_tokens() - grid.h.max(grid.w);
    assert_eq!(engine.rope_delta(), delta);
    let after = engine
        .snapshot_sequence()
        .expect("snapshot after the image");
    assert_eq!(after.rope_delta(), delta);
    engine.restore_sequence(&before).expect("restore");
    assert_eq!(engine.sequence_position(), k);
    assert_eq!(engine.rope_delta(), 0, "the offset in force at k");
    let got = engine.prefill_from_pos(&tail, k).expect("continue");
    let got_next = engine.decode_step(23, k + tail.len()).expect("decode");
    reference.prefill_from_pos(&lead, 0).expect("prefill");
    let want = reference.prefill_from_pos(&tail, k).expect("continue");
    let want_next = reference.decode_step(23, k + tail.len()).expect("decode");
    assert_eq!(bits(&got), bits(&want));
    assert_eq!(bits(&got_next), bits(&want_next));

    // A snapshot after an image restores that image's offset and the
    // decode it implies.
    engine.reset();
    engine.prefill_from_pos(&lead, 0).expect("prefill");
    engine
        .prefill_multimodal(&continuation, k)
        .expect("image continuation");
    let end = k + grid.n_tokens() + 1;
    let with_image = engine.snapshot_sequence().expect("snapshot");
    let first = engine.decode_step(40, end).expect("decode after the image");
    engine
        .prefill_from_pos(&tail, end + 1)
        .expect("text after it");
    engine.restore_sequence(&with_image).expect("restore");
    assert_eq!(engine.rope_delta(), delta);
    let again = engine.decode_step(40, end).expect("decode again");
    assert_eq!(bits(&again), bits(&first));
}

/// A restore whose positions were rewritten since the snapshot — the
/// sequence rolled back to an earlier point and continued past it with
/// an image — would rotate at the wrong offset, so both executors refuse
/// it with the same code, changing nothing.
#[test]
fn a_restore_across_a_rewritten_image_is_refused_on_either_executor() {
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    for mut engine in engines(&gguf) {
        let backend = engine.hybrid_backend();
        let lead = [10u32, 11, 12, 13];
        engine.prefill_from_pos(&lead, 0).expect("prefill");
        let early = engine.snapshot_sequence().expect("early snapshot");
        let text: Vec<u32> = (40..60).collect();
        engine.prefill_from_pos(&text, lead.len()).expect("text");
        let late = engine.snapshot_sequence().expect("late snapshot");
        assert_eq!(late.rope_delta(), 0);
        engine
            .restore_sequence(&early)
            .expect("restore the early one");
        let grid = GridSize { h: 2, w: 3 };
        let continuation = [
            PromptSegment::Image {
                rows: image(grid, 5).rows,
                grid,
            },
            PromptSegment::Text(vec![30]),
        ];
        engine
            .prefill_multimodal(&continuation, lead.len())
            .expect("image continuation");
        let end = engine.sequence_position();
        let more: Vec<u32> = (60..80).collect();
        engine
            .prefill_from_pos(&more, end)
            .expect("text past the late one");
        let position = engine.sequence_position();
        let err = engine
            .restore_sequence(&late)
            .expect_err("rewritten below the late snapshot");
        assert_eq!(
            engine_error_code(&err),
            Some("SNAPSHOT_MISMATCH"),
            "{backend:?}: {err}"
        );
        assert!(err.to_string().contains("M-RoPE offset"), "{err}");
        assert_eq!(engine.sequence_position(), position, "{backend:?}");
    }
}

/// A text-only prompt through the enum is exactly `generate`.
#[test]
fn text_prompts_are_unchanged_through_the_prompt_enum() {
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let mut engine = InferenceEngine::from_gguf(&gguf, greedy(), 7, MAX_SEQ).expect("engine");
    let tokens = vec![10u32, 11, 12, 13, 14];
    let direct = engine.generate(&tokens, 5).expect("generate");
    let via_enum = ChatPrompt::Text(tokens)
        .generate(&mut engine, 5)
        .expect("enum");
    assert_eq!(direct, via_enum);
}

#[test]
fn error_codes_and_client_fault_are_stable() {
    let cases: Vec<(MultimodalError, &str, bool)> = vec![
        (
            MultimodalError::VisionUnavailable,
            "vision_unavailable",
            true,
        ),
        (
            MultimodalError::NotAHybridModel {
                architecture: "qwen3".into(),
            },
            "NOT_A_HYBRID_MODEL",
            true,
        ),
        (
            MultimodalError::TooManyImages {
                images: 17,
                limit: MAX_IMAGES_PER_REQUEST,
            },
            "too_many_images",
            true,
        ),
        (
            MultimodalError::Image {
                index: 0,
                source: ImageInputError::Empty {
                    width: 0,
                    height: 1,
                },
            },
            "image_empty",
            true,
        ),
        (
            MultimodalError::Encode {
                index: 0,
                reason: "x".into(),
            },
            "image_encode_failed",
            false,
        ),
        (
            MultimodalError::ProjectorMismatch {
                projector_width: 96,
                model_width: 64,
                executor: executor_label(HybridBackend::Metal),
            },
            "projector_mismatch",
            false,
        ),
    ];
    for (err, code, client) in cases {
        assert_eq!(err.code(), code);
        assert_eq!(err.is_client_error(), client, "{code}");
        let runtime: RuntimeError = err.into();
        assert!(
            runtime.to_string().contains(&format!("[{code}]")),
            "{runtime}"
        );
    }
}
