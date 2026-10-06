//! Unit tests for `vision.rs` (sibling file, declared there via `#[path]`,
//! so `super` still names that module).

use super::*;
use crate::cli::cmd_run::{load_engine, EngineLoad};
use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_model::vision::{GridSize, VisionTokenIds};
use oxibonsai_runtime::config::RopeScalingMode;
use oxibonsai_runtime::engine_hybrid_gpu::HybridBackend;
use oxibonsai_runtime::sampling::{PenaltyParams, SamplingParams};
use oxibonsai_runtime::vision_prefill::{ChatPrompt, EncodedImage, MultimodalPrompt};
use oxibonsai_testkit::qwen35_fixture::{synthetic_qwen35_gguf, HIDDEN};

// ── The backend decision (a truth table over the three inputs) ──────────────

#[test]
fn only_auto_with_vision_and_an_engine_that_cannot_prefill_images_rebuilds() {
    use VisionBackendPlan::{Keep, RebuildOnCpu};
    for requested in [Backend::Auto, Backend::Cpu, Backend::Metal] {
        for wants_vision in [false, true] {
            for prefills_images in [false, true] {
                let expected = if requested == Backend::Auto && wants_vision && !prefills_images {
                    RebuildOnCpu
                } else {
                    Keep
                };
                assert_eq!(
                    plan_vision_backend(requested, wants_vision, prefills_images),
                    expected,
                    "requested {requested}, vision {wants_vision}, engine prefills images \
                     {prefills_images}"
                );
            }
        }
    }
}

/// The `INFO` line the downgrade logs says what happened and what the user
/// can pass instead — in flags and executor names, never in the names of
/// Rust items — and takes the executor from its argument.
#[test]
fn the_rebuild_reason_says_what_happened_and_what_to_pass_without_naming_code() {
    let reason = rebuild_reason("metal");
    assert!(reason.contains("--mmproj is loaded"), "{reason}");
    assert!(reason.contains("--backend auto"), "{reason}");
    assert!(reason.contains("metal runner"), "{reason}");
    assert!(reason.contains("cannot take image input yet"), "{reason}");
    assert!(
        reason.contains("using the CPU engine for this session"),
        "what happened: {reason}"
    );
    assert!(
        reason.contains("--backend metal to keep that runner and have image turns refused"),
        "the way to refuse instead: {reason}"
    );
    assert!(
        reason.contains("--backend cpu to skip building the metal engine first"),
        "the way to skip the extra build: {reason}"
    );

    // No Rust item reaches the user.
    assert!(!reason.contains("InferenceEngine::"), "{reason}");
    assert!(!reason.contains("multimodal_backend_is_c"), "{reason}");
    assert!(!reason.contains("::"), "{reason}");

    // The executor is the argument's, never a hard-coded one.
    let other = rebuild_reason("cuda");
    assert!(other.contains("cuda runner"), "{other}");
    assert!(
        other.contains("--backend cuda to keep that runner"),
        "{other}"
    );
    assert!(
        !other.to_ascii_lowercase().contains("metal"),
        "no hard-coded executor: {other}"
    );
}

#[cfg(feature = "server")]
#[test]
fn a_server_with_an_explicit_backend_that_cannot_serve_images_warns_at_startup() {
    assert_eq!(startup_backend_warning(Backend::Auto, false), None);
    assert_eq!(startup_backend_warning(Backend::Auto, true), None);
    assert_eq!(startup_backend_warning(Backend::Cpu, true), None);
    assert_eq!(startup_backend_warning(Backend::Metal, true), None);
    let warning = startup_backend_warning(Backend::Metal, false).expect("refusals ahead");
    assert!(warning.contains("--backend metal"), "{warning}");
    assert!(warning.contains("BACKEND_UNAVAILABLE"), "{warning}");
    assert!(
        warning.contains("--backend auto or --backend cpu"),
        "{warning}"
    );
}

// ── The synthetic hybrid, with a Metal device when the host has one ─────────

const MAX_SEQ: usize = 64;
const IDS: VisionTokenIds = VisionTokenIds {
    vision_start: 500,
    vision_end: 501,
    image_pad: 502,
};

fn load(backend: Backend, wants_vision: bool) -> EngineLoad {
    EngineLoad {
        params: SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        },
        seed: 7,
        max_seq_len: MAX_SEQ,
        backend,
        rope_scaling: RopeScalingMode::Auto,
        prefill_chunk: None,
        penalties: PenaltyParams::default(),
        min_p: 0.0,
        wants_vision,
        vision_resident_bytes: 0,
    }
}

/// One image (a 2 x 2 merged grid) between two text stretches.
fn image_prompt() -> MultimodalPrompt {
    let grid = GridSize { h: 2, w: 2 };
    let rows = (0..grid.n_tokens() * HIDDEN)
        .map(|i| ((i as u32).wrapping_mul(2_654_435_761) >> 8) as f32 / 16_777_216.0 - 0.5)
        .collect();
    let image = EncodedImage {
        rows,
        grid,
        source: (64, 64),
    };
    let tokens = vec![
        10,
        11,
        IDS.vision_start,
        IDS.image_pad,
        IDS.vision_end,
        12,
        13,
    ];
    MultimodalPrompt::new(tokens, vec![image], IDS).expect("the prompt splices")
}

/// Whether this host resolves the fixture to the Metal runner under
/// `--backend auto` — the case the rebuild exists for. (A host without a
/// Metal device resolves it to the CPU model, and nothing needs rebuilding.)
fn host_resolves_auto_to_metal(gguf: &GgufFile<'_>) -> bool {
    auto_resolution(gguf).0 == Some(HybridBackend::Metal)
}

/// What `--backend auto` resolves the fixture to when no projector is
/// loaded: the executor, and whether that engine prefills image rows
/// (`prefills_images`) — the reference the projector cases are held against,
/// so no case here names a backend that can or cannot prefill them.
fn auto_resolution(gguf: &GgufFile<'_>) -> (Option<HybridBackend>, bool) {
    let engine = load_engine(gguf, &load(Backend::Auto, false), 0).expect("an auto engine");
    (engine.hybrid_backend(), engine.prefills_images())
}

/// `--mmproj` with `--backend auto`: the engine that comes back prefills
/// image rows and serves a real image turn. When the executor `auto` resolves
/// to cannot prefill image rows that engine is the rebuilt CPU model; when it
/// can — both hybrid executors can — it is the engine `auto` resolved to,
/// untouched. The engine's own predicate decides — nothing here names a
/// backend.
#[test]
fn auto_with_a_projector_yields_an_engine_that_serves_an_image_turn() {
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("the fixture parses");
    let (auto_executor, auto_prefills_images) = auto_resolution(&gguf);

    let mut engine = load_engine(&gguf, &load(Backend::Auto, true), 0).expect("a vision engine");
    assert!(
        engine.prefills_images(),
        "the engine a vision command runs prefills image rows"
    );
    let expected_executor = if auto_prefills_images {
        auto_executor
    } else {
        Some(HybridBackend::Cpu)
    };
    assert_eq!(
        engine.hybrid_backend(),
        expected_executor,
        "kept as auto resolved it ({auto_executor:?}, prefills image rows: \
         {auto_prefills_images}), else rebuilt on the CPU model"
    );
    assert!(
        require_image_capable_engine(&engine).is_ok(),
        "nothing to refuse"
    );

    // A successful image turn: the multimodal prefill runs and the engine
    // generates from it.
    let prompt = image_prompt();
    let logits = engine
        .prefill_multimodal(&prompt.segments(), 0)
        .expect("the image rows prefill");
    assert!(logits.iter().all(|v| v.is_finite()));
    let tokens = ChatPrompt::Multimodal(prompt)
        .generate(&mut engine, 4)
        .expect("the engine generates from the image prompt");
    assert!(!tokens.is_empty(), "an answer of at least one token");
}

/// An explicit `--backend metal` is never overridden: with a projector the
/// engine stays the Metal one. Were that runner unable to prefill image rows
/// the image turn would keep the typed `BACKEND_UNAVAILABLE` refusal, raised
/// before any image is encoded; the refusal follows the engine's own
/// predicate, and the runner does prefill them, so the turn is served.
#[test]
fn an_explicit_metal_backend_with_a_projector_is_never_overridden() {
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("the fixture parses");
    if !host_resolves_auto_to_metal(&gguf) {
        // No Metal runner on this host: `--backend metal` cannot build an
        // engine at all (a typed `HybridGpuBackendUnsupported`), so there is
        // no Metal engine to keep.
        let refused = load_engine(&gguf, &load(Backend::Metal, true), 0);
        assert!(
            refused.is_err(),
            "no Metal device: an explicit metal is refused"
        );
        return;
    }
    let mut engine = load_engine(&gguf, &load(Backend::Metal, true), 0).expect("a Metal engine");
    assert_eq!(
        engine.hybrid_backend(),
        Some(HybridBackend::Metal),
        "not rebuilt"
    );
    if engine.prefills_images() {
        // The runner prefills image rows: there is nothing to refuse, and an
        // image turn runs on the runner itself.
        assert!(require_image_capable_engine(&engine).is_ok());
        let logits = engine
            .prefill_multimodal(&image_prompt().segments(), 0)
            .expect("the runner prefills the image rows");
        assert!(logits.iter().all(|v| v.is_finite()));
        assert_eq!(engine.hybrid_backend(), Some(HybridBackend::Metal));
        return;
    }

    let err = require_image_capable_engine(&engine).expect_err("a runner cannot prefill images");
    let text = err.to_string();
    assert!(text.contains("BACKEND_UNAVAILABLE"), "{text}");
    assert!(text.contains("--backend cpu"), "names the way out: {text}");

    // The engine's own refusal is the same typed one.
    let refusal = engine
        .prefill_multimodal(&image_prompt().segments(), 0)
        .expect_err("the runner refuses image rows");
    assert_eq!(
        oxibonsai_runtime::engine_seam::engine_error_code(&refusal),
        Some("BACKEND_UNAVAILABLE"),
        "{refusal}"
    );
}

/// No projector: `--backend auto` is exactly what it was — the Metal runner
/// on a Metal host — and a text-only run never pays for a second build.
#[test]
fn auto_without_a_projector_is_unchanged() {
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("the fixture parses");
    let metal_host = host_resolves_auto_to_metal(&gguf);
    let engine = load_engine(&gguf, &load(Backend::Auto, false), 0).expect("an auto engine");
    let expected = if metal_host {
        HybridBackend::Metal
    } else {
        HybridBackend::Cpu
    };
    assert_eq!(engine.hybrid_backend(), Some(expected));
}

/// An explicit `--backend cpu` with a projector is what it says.
#[test]
fn an_explicit_cpu_backend_with_a_projector_is_kept() {
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("the fixture parses");
    let engine = load_engine(&gguf, &load(Backend::Cpu, true), 0).expect("a CPU engine");
    assert_eq!(engine.hybrid_backend(), Some(HybridBackend::Cpu));
    assert!(require_image_capable_engine(&engine).is_ok());
}

// ── The room a prompt leaves ────────────────────────────────────────────────

#[test]
fn a_prompt_that_fits_with_room_to_generate_is_accepted() {
    check_generation_room(10, 4, 64, 0).expect("room for the whole request");
    // A prompt that fits but leaves less than `max_tokens` is clamped later,
    // never refused: the CLI treats an over-large `--max-tokens` that way
    // everywhere.
    check_generation_room(60, 256, 64, 0).expect("clamped, not refused");
    check_generation_room(63, 256, 64, 1).expect("one position left is room");
}

#[test]
fn a_prompt_that_alone_overflows_the_context_is_refused_naming_both_numbers() {
    let text = check_generation_room(70, 4, 64, 0)
        .expect_err("70 > 64")
        .to_string();
    assert!(text.contains("sequence length 70"), "{text}");
    assert!(text.contains("max context 64"), "{text}");
    assert!(text.contains("shorten it or raise --ctx"), "{text}");
    assert!(
        !text.contains("image"),
        "a text prompt has no images: {text}"
    );

    let images = check_generation_room(70, 4, 64, 2)
        .expect_err("70 > 64")
        .to_string();
    assert!(
        images.contains("the prompt with its image rows"),
        "{images}"
    );
    assert!(images.contains("--image-max-tokens"), "{images}");
}

/// The prompt fills the window exactly: no position is left for a single
/// generated token, so prefilling it (minutes for an image prompt on the
/// 27B) could only end in an empty answer.
#[test]
fn a_prompt_that_fills_the_context_exactly_leaves_nothing_to_generate_into() {
    for images in [0usize, 1] {
        let text = check_generation_room(64, 1, 64, images)
            .expect_err("64 of 64 leaves no room")
            .to_string();
        assert!(text.contains("fills max context 64 exactly"), "{text}");
        assert!(text.contains("generate 1 token(s)"), "{text}");
    }
    // Asking for nothing to be generated is not an error of the prompt.
    check_generation_room(64, 0, 64, 0).expect("no tokens requested");
}

// ── Serving images: the timeout guidance, the served id, the backend ─────────

#[cfg(feature = "server")]
mod serving {
    use super::*;
    use crate::cli::bonsai2::tests::bonsai2_vocabulary;
    use crate::cli::bonsai2::{ImageSourceFlags, VisionRequest, MEDIA_PATH_ENV};
    use crate::cli::cmd_serve::{
        build_serving_pool, embedding_backend, served_input_ceiling, served_model_id_for,
        vision_timeout_warning, ServingPoolLoad, VISION_REQUEST_TIMEOUT_HINT_MS,
    };
    use crate::cli::util::build_sampling_params;
    use crate::cli::util::test_env::{self, EnvVarGuard};
    use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue};

    // ── The projector a server loads ─────────────────────────────────────

    /// A `OXI_MEDIA_PATH` left in a shell — here naming a directory that is
    /// gone — must not stop a text-only server: without `--mmproj` nothing
    /// consults the media directory. With a projector the same environment is
    /// an error, naming the variable.
    #[test]
    fn a_stale_media_path_environment_does_not_stop_a_text_only_server() {
        let _env = test_env::lock();
        let missing = std::env::temp_dir().join("oxibonsai-no-such-media-directory");
        let _stale = EnvVarGuard::set(MEDIA_PATH_ENV, &missing);

        let text_only = serve_vision_service(
            &VisionRequest::default(),
            &ImageSourceFlags::default(),
            "qwen35",
            &bonsai2_vocabulary(),
        )
        .expect("a text-only server needs no media directory");
        assert!(text_only.is_none());

        let dir = crate::cli::test_fixtures::scratch_dir("serve_vision_stale_env");
        let with_projector = VisionRequest {
            mmproj: Some(crate::cli::bonsai2::tests::synthetic_projector(&dir)),
            ..VisionRequest::default()
        };
        let msg = serve_vision_service(
            &with_projector,
            &ImageSourceFlags::default(),
            "qwen35",
            &bonsai2_vocabulary(),
        )
        .map(|_| ())
        .expect_err("a server that serves images needs a real media directory")
        .to_string();
        assert!(msg.contains("OXI_MEDIA_PATH"), "{msg}");
        assert!(msg.contains("cannot be resolved"), "{msg}");

        // A flag that is given takes the stale environment out of the picture.
        let media = dir.join("media");
        std::fs::create_dir_all(&media).expect("media dir");
        let with_flag = ImageSourceFlags {
            allow_image_url_fetch: false,
            media_path: Some(media.to_string_lossy().into_owned()),
            ..ImageSourceFlags::default()
        };
        let service =
            serve_vision_service(&with_projector, &with_flag, "qwen35", &bonsai2_vocabulary())
                .expect("the flag names a real directory")
                .expect("a projector was requested");
        assert_eq!(
            service.token_ids(),
            oxibonsai_model::vision::VisionTokenIds::BONSAI2
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── The model's vocabulary ───────────────────────────────────────────

    /// A server that loads a projector holds the image tokens of the model's
    /// vocabulary against the ids the splice uses: a vocabulary that lacks
    /// them refuses to start, with the typed error. A text-only server never
    /// reads the vocabulary.
    #[test]
    fn a_server_whose_model_vocabulary_disagrees_with_the_splice_refuses_to_start() {
        use crate::cli::bonsai2::tests::vocabulary_tokens;
        use crate::cli::bonsai2::{ModelVocabulary, VisionMarkerError};

        let _env = test_env::lock();
        let dir = crate::cli::test_fixtures::scratch_dir("serve_vision_vocabulary");
        let media = dir.join("media");
        std::fs::create_dir_all(&media).expect("media dir");
        let sources = ImageSourceFlags {
            allow_image_url_fetch: false,
            media_path: Some(media.to_string_lossy().into_owned()),
            ..ImageSourceFlags::default()
        };
        let with_projector = VisionRequest {
            mmproj: Some(crate::cli::bonsai2::tests::synthetic_projector(&dir)),
            ..VisionRequest::default()
        };
        let lacking = vocabulary_tokens(600, &[]);
        let err = serve_vision_service(
            &with_projector,
            &sources,
            "qwen35",
            &ModelVocabulary::from_tokens(&lacking),
        )
        .map(|_| ())
        .expect_err("the vocabulary has no image tokens");
        let typed = err
            .downcast_ref::<VisionMarkerError>()
            .expect("the typed vocabulary refusal");
        assert_eq!(typed.problems.len(), 3, "{err}");
        assert_eq!(typed.vocabulary_size, 600);

        // The same projector over a vocabulary that agrees serves.
        assert!(
            serve_vision_service(&with_projector, &sources, "qwen35", &bonsai2_vocabulary())
                .expect("a matching vocabulary starts")
                .is_some()
        );
        // No projector: the vocabulary is never consulted.
        assert!(serve_vision_service(
            &VisionRequest::default(),
            &sources,
            "qwen35",
            &ModelVocabulary::from_tokens(&lacking),
        )
        .expect("a text-only server")
        .is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── The start-up timeout warning ─────────────────────────────────────

    /// A server that loaded a projector and runs with the default 60 s
    /// budget says what an image request costs and which flag to raise.
    #[test]
    fn a_short_request_timeout_with_a_projector_warns_with_the_measured_costs() {
        let warning = vision_timeout_warning(60_000).expect("60 s is below the guidance");
        assert!(warning.contains("--mmproj is loaded"), "{warning}");
        assert!(
            warning.contains("--request-timeout-ms is 60000 ms (60 s)"),
            "names the value it saw: {warning}"
        );
        assert!(
            warning.contains("tens of seconds to minutes"),
            "the order of magnitude: {warning}"
        );
        assert!(
            warning.contains("67 rows 54.9 s"),
            "a measured figure: {warning}"
        );
        assert!(
            warning.contains("75 rows 32.6 s"),
            "a measured figure: {warning}"
        );
        // The two figures come from separate runs at different host load
        // (the larger prompt was the faster run): the warning says so.
        assert!(
            warning.contains("measured on separate runs at different host load"),
            "{warning}"
        );
        assert!(warning.contains("order of magnitude only"), "{warning}");
        assert!(
            warning.contains("about 5 s"),
            "the vision encode: {warning}"
        );
        assert!(
            warning.contains("Raise it with --request-timeout-ms 300000"),
            "the flag to raise, and to what: {warning}"
        );
        assert!(warning.contains("naming the stage"), "{warning}");
        // The cost depends on the executor the server decodes on, and the
        // warning says so instead of assuming one.
        assert!(
            warning.contains("on the executor the server decodes on"),
            "{warning}"
        );
        assert!(warning.contains("Metal hybrid runner"), "{warning}");
        assert!(warning.contains("CPU model"), "{warning}");
    }

    #[test]
    fn the_warning_stops_at_the_guidance_threshold() {
        assert!(vision_timeout_warning(0).is_some());
        assert!(vision_timeout_warning(VISION_REQUEST_TIMEOUT_HINT_MS - 1).is_some());
        assert_eq!(vision_timeout_warning(VISION_REQUEST_TIMEOUT_HINT_MS), None);
        assert_eq!(vision_timeout_warning(600_000), None);
        assert_eq!(vision_timeout_warning(u64::MAX), None);
    }

    // ── The prompt ceiling ───────────────────────────────────────────────

    /// `serve`'s prompt ceiling is the KV window the replicas were actually
    /// built with — the number the request budget also checks — never above
    /// the window that was asked for.
    #[test]
    fn the_prompt_ceiling_is_the_smaller_of_the_requested_and_the_built_window() {
        assert_eq!(served_input_ceiling(8192, 8192), 8192, "the usual case");
        assert_eq!(
            served_input_ceiling(8192, 4096),
            4096,
            "a Metal window clamped below the request"
        );
        assert_eq!(served_input_ceiling(4096, 8192), 4096);
    }

    /// The router's budget check and `serve`'s ceiling agree: an engine built
    /// at a window of 64 serves prompts up to that window only.
    #[test]
    fn the_ceiling_agrees_with_the_window_the_engine_reports() {
        let bytes = synthetic_qwen35_gguf();
        let gguf = GgufFile::parse(&bytes).expect("the fixture parses");
        let engine = oxibonsai_runtime::InferenceEngine::from_gguf_with_backend(
            &gguf,
            SamplingParams::default(),
            42,
            64,
            Backend::Cpu,
        )
        .expect("a CPU hybrid engine");
        assert_eq!(engine.max_context(), 64);
        assert_eq!(served_input_ceiling(64, engine.max_context()), 64);
        assert_eq!(served_input_ceiling(8192, engine.max_context()), 64);
    }

    // ── The embedder's backend ───────────────────────────────────────────

    /// A hybrid model's embedder is built on the CPU model whatever
    /// `--backend` asked for, and the start-up line says so; a dense model's
    /// follows the flag.
    #[test]
    fn a_hybrid_embedder_is_reported_on_the_cpu_whatever_backend_was_requested() {
        for requested in [Backend::Auto, Backend::Cpu, Backend::Metal] {
            assert_eq!(embedding_backend(true, requested), Backend::Cpu);
            assert_eq!(embedding_backend(false, requested), requested);
        }
    }

    // ── The id the model is served under ─────────────────────────────────

    fn gguf_named(name: Option<&str>) -> Vec<u8> {
        let mut writer = GgufWriter::new();
        writer.add_metadata(
            "general.architecture",
            MetadataWriteValue::Str("qwen35".to_string()),
        );
        if let Some(name) = name {
            writer.add_metadata("general.name", MetadataWriteValue::Str(name.to_string()));
        }
        writer.to_bytes().expect("serialize metadata")
    }

    #[test]
    fn the_27b_placeholder_name_is_served_under_the_file_stem() {
        let bytes = gguf_named(Some("Hf"));
        let gguf = GgufFile::parse(&bytes).expect("parses");
        assert_eq!(
            served_model_id_for(&gguf, "models/Ternary-Bonsai-2-27B-PQ2_0.gguf").as_deref(),
            Some("Ternary-Bonsai-2-27B-PQ2_0")
        );
        // Whatever the extension and directory.
        assert_eq!(
            served_model_id_for(&gguf, "PQ2.other.gguf").as_deref(),
            Some("PQ2.other")
        );
    }

    #[test]
    fn a_real_general_name_is_served_as_the_file_spells_it() {
        let bytes = gguf_named(Some("Ternary-Bonsai-2-27B"));
        let gguf = GgufFile::parse(&bytes).expect("parses");
        assert_eq!(
            served_model_id_for(&gguf, "models/renamed.gguf").as_deref(),
            Some("Ternary-Bonsai-2-27B")
        );
    }

    /// A file with no `general.name` at all is named by its stem (not by the
    /// architecture tag a loader would substitute), and a name that is
    /// unusable with no stem to fall back on leaves the router's own name.
    #[test]
    fn a_missing_name_falls_back_to_the_stem_and_nothing_usable_to_none() {
        let bytes = gguf_named(None);
        let gguf = GgufFile::parse(&bytes).expect("parses");
        assert_eq!(
            served_model_id_for(&gguf, "models/Bonsai-8B.gguf").as_deref(),
            Some("Bonsai-8B")
        );
        let placeholder = gguf_named(Some("gguf"));
        let gguf = GgufFile::parse(&placeholder).expect("parses");
        assert_eq!(served_model_id_for(&gguf, ""), None);
        assert_eq!(served_model_id_for(&gguf, "/"), None);
    }

    // ── The backend of a serving pool ────────────────────────────────────

    fn leaked_synthetic() -> &'static GgufFile<'static> {
        let bytes: &'static [u8] = Box::leak(synthetic_qwen35_gguf().into_boxed_slice());
        Box::leak(Box::new(
            GgufFile::parse(bytes).expect("the fixture parses"),
        ))
    }

    fn pool_load(params: &SamplingParams, wants_vision: bool) -> ServingPoolLoad<'_> {
        ServingPoolLoad {
            params,
            seed: 42,
            max_seq_len: 64,
            requested_pool_size: Some(1),
            rope_scaling: RopeScalingMode::Auto,
            wants_vision,
            hybrid: oxibonsai_runtime::engine_hybrid_gpu::HybridLoadOptions::default(),
        }
    }

    async fn first_replica(
        built: &oxibonsai_runtime::engine_pool::PoolBuild,
    ) -> (Option<HybridBackend>, bool) {
        let lease = built.pool.acquire().await.expect("a replica");
        (lease.hybrid_backend(), lease.prefills_images())
    }

    /// `--mmproj` with `--backend auto`: the pool that comes back prefills
    /// image rows. When the executor `auto` resolves to cannot, the pool is
    /// rebuilt on the CPU model and `backend` reports the CPU; when it can —
    /// both hybrid executors can — the pool is the one `auto` resolved to,
    /// unchanged.
    #[tokio::test]
    async fn auto_with_a_projector_serves_from_a_pool_that_prefills_images() {
        let gguf = leaked_synthetic();
        let params = build_sampling_params(0.0, 40, 0.9, 1.0);

        // What `auto` resolves to with no projector: the reference.
        let (auto, _) = build_serving_pool(gguf, &pool_load(&params, false), Backend::Auto)
            .await
            .expect("an auto pool");
        let (auto_executor, auto_prefills) = first_replica(&auto).await;
        drop(auto);

        let (built, backend) = build_serving_pool(gguf, &pool_load(&params, true), Backend::Auto)
            .await
            .expect("a vision pool");
        let (executor, prefills_images) = first_replica(&built).await;
        assert!(prefills_images, "a vision pool prefills image rows");
        if auto_prefills {
            assert_eq!(
                (backend, executor),
                (Backend::Auto, auto_executor),
                "nothing to rebuild"
            );
        } else {
            assert_eq!(
                (backend, executor),
                (Backend::Cpu, Some(HybridBackend::Cpu)),
                "rebuilt on the CPU model"
            );
        }
    }

    /// An explicit backend is never overridden, and `auto` without a
    /// projector is what it always was.
    #[tokio::test]
    async fn explicit_backends_and_projector_less_auto_are_left_alone() {
        let gguf = leaked_synthetic();
        let params = build_sampling_params(0.0, 40, 0.9, 1.0);

        let (built, backend) = build_serving_pool(gguf, &pool_load(&params, true), Backend::Cpu)
            .await
            .expect("an explicit CPU pool");
        assert_eq!(backend, Backend::Cpu);
        assert_eq!(
            first_replica(&built).await,
            (Some(HybridBackend::Cpu), true)
        );
        drop(built);

        // What `auto` resolves to on this host, with no projector: the
        // reference the other two cases are held against.
        let (auto, backend) = build_serving_pool(gguf, &pool_load(&params, false), Backend::Auto)
            .await
            .expect("an auto pool");
        assert_eq!(
            backend,
            Backend::Auto,
            "no projector: nothing decided anything"
        );
        let (auto_executor, auto_prefills) = first_replica(&auto).await;
        drop(auto);
        let metal_host = auto_executor == Some(HybridBackend::Metal);

        if metal_host {
            // `--backend metal` with a projector stays on the runner, whatever
            // it can prefill: image requests get the runner's own answer (an
            // image turn on the runner, or its typed refusal were it unable
            // to prefill image rows), and the pool says so.
            let (built, backend) =
                build_serving_pool(gguf, &pool_load(&params, true), Backend::Metal)
                    .await
                    .expect("an explicit Metal pool");
            assert_eq!(backend, Backend::Metal);
            assert_eq!(
                first_replica(&built).await,
                (Some(HybridBackend::Metal), auto_prefills)
            );
        }
    }

    /// The pool is built under its hybrid options: every replica takes the
    /// prefill chunk (the CPU model holds it, and a Metal runner's calls are
    /// sized for it) and a Metal replica's KV window counts the vision
    /// tower's resident bytes — whichever executor this host resolves `auto`
    /// to, the expectation is read off the replica itself.
    #[tokio::test]
    async fn every_replica_is_built_under_the_pools_hybrid_options() {
        let gguf = leaked_synthetic();
        let params = build_sampling_params(0.0, 40, 0.9, 1.0);
        let options = oxibonsai_runtime::engine_hybrid_gpu::HybridLoadOptions {
            vision_resident_bytes: 3 << 20,
            prefill_chunk: Some(24),
        };
        let load = ServingPoolLoad {
            hybrid: options,
            ..pool_load(&params, true)
        };
        let (built, _) = build_serving_pool(gguf, &load, Backend::Auto)
            .await
            .expect("a pool under the options");
        let lease = built.pool.acquire().await.expect("a replica");
        assert_eq!(
            lease.hybrid_model().map(|m| m.prefill_chunk()),
            Some(24),
            "the CPU model holds the chunk"
        );
        assert_eq!(
            lease.prefill_chunk_in_effect(),
            24,
            "the replica's executor takes calls of the chunk"
        );
        match lease.hybrid_metal_window() {
            Some(window) => assert_eq!(window.vision_resident_bytes, 3 << 20),
            None => assert_eq!(lease.hybrid_backend(), Some(HybridBackend::Cpu)),
        }
        drop(lease);
        // Outside the build the options are gone: nothing leaks to this
        // thread's later engines.
        assert_eq!(
            oxibonsai_runtime::engine_hybrid_gpu::HybridLoadScope::active(),
            oxibonsai_runtime::engine_hybrid_gpu::HybridLoadOptions::default()
        );
    }
}
