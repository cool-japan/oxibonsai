//! Tests for the image tokens against the model's vocabulary
//! ([`check_vision_markers`], [`VisionRequest::load_service`]): sibling of
//! `bonsai2_tests.rs`, declared in `bonsai2.rs` via `#[path]`, so `super`
//! still names that module.

use super::tests::{
    bonsai2_vocabulary, qwen35_gguf_with_vocabulary, synthetic_projector, vocabulary_tokens,
    BONSAI2_MARKERS,
};
use super::*;
use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::MetadataValue;
use oxibonsai_model::vision::VisionTokenIds;

fn scratch(tag: &str) -> std::path::PathBuf {
    crate::cli::test_fixtures::scratch_dir(&format!("bonsai2_markers_{tag}"))
}

fn with_projector(mmproj: &str) -> VisionRequest {
    VisionRequest {
        mmproj: Some(mmproj.to_string()),
        ..VisionRequest::default()
    }
}

#[test]
fn the_bonsai2_vocabulary_holds_the_image_tokens_where_the_splice_expects_them() {
    let vocabulary = bonsai2_vocabulary();
    assert_eq!(vocabulary.token_count(), 248_057);
    check_vision_markers("mmproj.gguf", VisionTokenIds::BONSAI2, &vocabulary)
        .expect("the splice's ids are the vocabulary's ids");
}

/// A vocabulary that lacks all three tokens: every one is named with the id
/// the splice would have used, none with a vocabulary id.
#[test]
fn a_vocabulary_lacking_the_image_tokens_is_refused_naming_each_with_its_splice_id() {
    let tokens = vocabulary_tokens(600, &[("<|im_start|>", 3), ("<|im_end|>", 4)]);
    let vocabulary = ModelVocabulary::from_tokens(&tokens);
    let err = check_vision_markers(
        "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf",
        VisionTokenIds::BONSAI2,
        &vocabulary,
    )
    .expect_err("no image token is in this vocabulary");

    assert_eq!(err.code(), "vision_vocabulary_mismatch");
    assert_eq!(err.vocabulary_size, 600);
    assert_eq!(
        err.problems,
        vec![
            MarkerProblem {
                token: VISION_START_TOKEN,
                splice_id: 248_053,
                vocabulary_id: None,
            },
            MarkerProblem {
                token: VISION_END_TOKEN,
                splice_id: 248_054,
                vocabulary_id: None,
            },
            MarkerProblem {
                token: IMAGE_PAD_TOKEN,
                splice_id: 248_056,
                vocabulary_id: None,
            },
        ]
    );
    let text = err.to_string();
    assert!(
        text.starts_with("[vision_vocabulary_mismatch] --mmproj "),
        "{text}"
    );
    assert!(
        text.contains("Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"),
        "{text}"
    );
    assert!(text.contains("(600 tokens)"), "{text}");
    for (token, id) in BONSAI2_MARKERS {
        assert!(
            text.contains(&format!(
                "{token} is missing from the model's vocabulary (the image splice uses id {id})"
            )),
            "{token} named with its splice id: {text}"
        );
    }
    assert!(text.contains("or drop --mmproj"), "{text}");
}

/// A token that sits at another id in the vocabulary: named with BOTH ids,
/// and only that token — the two that are where the splice expects them are
/// not mentioned.
#[test]
fn an_image_token_at_another_id_is_refused_naming_both_ids() {
    let tokens = vocabulary_tokens(
        248_200,
        &[
            (VISION_START_TOKEN, 248_100),
            (VISION_END_TOKEN, 248_054),
            (IMAGE_PAD_TOKEN, 248_056),
        ],
    );
    let err = check_vision_markers(
        "m.gguf",
        VisionTokenIds::BONSAI2,
        &ModelVocabulary::from_tokens(&tokens),
    )
    .expect_err("<|vision_start|> is at 248100, not 248053");

    assert_eq!(
        err.problems,
        vec![MarkerProblem {
            token: VISION_START_TOKEN,
            splice_id: 248_053,
            vocabulary_id: Some(248_100),
        }]
    );
    let text = err.to_string();
    assert!(
        text.contains(
            "<|vision_start|> is id 248100 in the model's vocabulary but the image splice uses \
             id 248053"
        ),
        "{text}"
    );
    assert!(!text.contains(VISION_END_TOKEN), "{text}");
    assert!(!text.contains(IMAGE_PAD_TOKEN), "{text}");
}

/// Every wrong token is reported in one refusal, not one per attempt.
#[test]
fn every_misplaced_or_missing_image_token_is_reported_at_once() {
    let tokens = vocabulary_tokens(
        248_300,
        &[(VISION_START_TOKEN, 248_100), (IMAGE_PAD_TOKEN, 248_056)],
    );
    let err = check_vision_markers(
        "m.gguf",
        VisionTokenIds::BONSAI2,
        &ModelVocabulary::from_tokens(&tokens),
    )
    .expect_err("one misplaced, one missing");
    assert_eq!(
        err.problems.iter().map(|p| p.token).collect::<Vec<_>>(),
        vec![VISION_START_TOKEN, VISION_END_TOKEN],
        "in splice order, the correct one left out"
    );
    let text = err.to_string();
    assert!(
        text.contains("id 248100 in the model's vocabulary"),
        "{text}"
    );
    assert!(
        text.contains("<|vision_end|> is missing from the model's vocabulary"),
        "{text}"
    );
}

/// The ids to compare with are the splice layer's — whatever they are — and
/// a token that ALSO appears at some other id is fine so long as it is where
/// the splice expects it.
#[test]
fn the_check_holds_the_vocabulary_against_the_ids_it_is_given() {
    let ids = VisionTokenIds {
        vision_start: 500,
        vision_end: 501,
        image_pad: 502,
    };
    let mut tokens = vocabulary_tokens(
        512,
        &[
            (VISION_START_TOKEN, 500),
            (VISION_END_TOKEN, 501),
            (IMAGE_PAD_TOKEN, 502),
        ],
    );
    tokens[7] = MetadataValue::String(IMAGE_PAD_TOKEN.to_string());
    check_vision_markers("m.gguf", ids, &ModelVocabulary::from_tokens(&tokens))
        .expect("every token is at its id (one has a second home)");
    // The Bonsai 2 constants are NOT this vocabulary's ids.
    assert!(check_vision_markers(
        "m.gguf",
        VisionTokenIds::BONSAI2,
        &ModelVocabulary::from_tokens(&tokens)
    )
    .is_err());
}

/// A GGUF with no `tokenizer.ggml.tokens` has nothing to verify the ids
/// against: refused, the message saying the vocabulary is what is missing.
#[test]
fn a_model_without_a_vocabulary_is_refused_naming_the_missing_array() {
    let err = check_vision_markers(
        "m.gguf",
        VisionTokenIds::BONSAI2,
        &ModelVocabulary::from_tokens(&[]),
    )
    .expect_err("nothing to check against");
    assert_eq!(err.vocabulary_size, 0);
    assert_eq!(err.problems.len(), 3);
    let text = err.to_string();
    assert!(
        text.contains("carries no vocabulary (tokenizer.ggml.tokens)"),
        "{text}"
    );
    assert!(!text.contains("(0 tokens)"), "{text}");
    for (token, id) in BONSAI2_MARKERS {
        assert!(
            text.contains(token) && text.contains(&id.to_string()),
            "{text}"
        );
    }
}

/// The vocabulary is read from the GGUF's own `tokenizer.ggml.tokens` — a
/// qwen35 GGUF that carries the image tokens at the splice's ids passes, one
/// whose ids are shifted by a token or that carries no vocabulary at all is
/// refused.
#[test]
fn a_qwen35_gguf_vocabulary_is_read_from_the_file() {
    let good = vocabulary_tokens(248_057, &BONSAI2_MARKERS);
    let bytes = qwen35_gguf_with_vocabulary(Some(&good));
    let gguf = GgufFile::parse(&bytes).expect("parse the synthetic qwen35 GGUF");
    let vocabulary = ModelVocabulary::of_gguf(&gguf);
    assert_eq!(vocabulary.token_count(), 248_057);
    check_vision_markers("m.gguf", VisionTokenIds::BONSAI2, &vocabulary)
        .expect("the file's vocabulary has the tokens at the splice's ids");

    let shifted = vocabulary_tokens(
        248_058,
        &[
            (VISION_START_TOKEN, 248_054),
            (VISION_END_TOKEN, 248_055),
            (IMAGE_PAD_TOKEN, 248_057),
        ],
    );
    let bytes = qwen35_gguf_with_vocabulary(Some(&shifted));
    let gguf = GgufFile::parse(&bytes).expect("parse");
    let err = check_vision_markers(
        "m.gguf",
        VisionTokenIds::BONSAI2,
        &ModelVocabulary::of_gguf(&gguf),
    )
    .expect_err("every id is one off");
    assert_eq!(
        err.problems
            .iter()
            .map(|p| (p.token, p.splice_id, p.vocabulary_id))
            .collect::<Vec<_>>(),
        vec![
            (VISION_START_TOKEN, 248_053, Some(248_054)),
            (VISION_END_TOKEN, 248_054, Some(248_055)),
            (IMAGE_PAD_TOKEN, 248_056, Some(248_057)),
        ]
    );

    let bytes = qwen35_gguf_with_vocabulary(None);
    let gguf = GgufFile::parse(&bytes).expect("parse");
    let vocabulary = ModelVocabulary::of_gguf(&gguf);
    assert_eq!(vocabulary.token_count(), 0);
    assert_eq!(
        check_vision_markers("m.gguf", VisionTokenIds::BONSAI2, &vocabulary)
            .expect_err("no vocabulary")
            .vocabulary_size,
        0
    );
}

/// `load_service` loads the projector for a synthetic qwen35 model whose
/// vocabulary agrees with the splice's ids, and refuses — with the typed
/// error, naming each token and both ids — the same projector for a
/// vocabulary that lacks the tokens or places one elsewhere.
#[test]
fn the_projector_loads_only_for_a_model_whose_vocabulary_matches_the_splice() {
    let dir = scratch("projector_vocabulary");
    let mmproj = synthetic_projector(&dir);
    let request = with_projector(&mmproj);

    // The synthetic qwen35 model carries the tokens where the splice expects.
    let good = vocabulary_tokens(248_057, &BONSAI2_MARKERS);
    let bytes = qwen35_gguf_with_vocabulary(Some(&good));
    let gguf = GgufFile::parse(&bytes).expect("parse the synthetic qwen35 GGUF");
    let service = request
        .load_service(
            "qwen35",
            &ModelVocabulary::of_gguf(&gguf),
            cli_image_policy(),
        )
        .expect("a matching vocabulary loads")
        .expect("a projector was requested");
    assert_eq!(service.token_ids(), VisionTokenIds::BONSAI2);

    // A vocabulary lacking the tokens: refused before the service is handed out.
    let lacking = vocabulary_tokens(600, &[]);
    let err = request
        .load_service(
            "qwen35",
            &ModelVocabulary::from_tokens(&lacking),
            cli_image_policy(),
        )
        .map(|_| ())
        .expect_err("the vocabulary has no image tokens");
    let typed = err
        .downcast_ref::<VisionMarkerError>()
        .expect("the refusal is the typed marker error");
    assert_eq!(typed.mmproj, mmproj);
    assert_eq!(typed.problems.len(), 3);
    let text = err.to_string();
    assert!(text.contains(&mmproj), "names the projector: {text}");
    for (token, id) in BONSAI2_MARKERS {
        assert!(
            text.contains(token) && text.contains(&id.to_string()),
            "{text}"
        );
    }

    // One token at another id: that token, with both ids.
    let moved = vocabulary_tokens(
        248_200,
        &[
            (VISION_START_TOKEN, 248_053),
            (VISION_END_TOKEN, 248_054),
            (IMAGE_PAD_TOKEN, 248_150),
        ],
    );
    let err = request
        .load_service(
            "qwen35",
            &ModelVocabulary::from_tokens(&moved),
            cli_image_policy(),
        )
        .map(|_| ())
        .expect_err("<|image_pad|> is at 248150");
    let text = err.to_string();
    assert!(
        text.contains("<|image_pad|> is id 248150 in the model's vocabulary but the image splice uses id 248056"),
        "{text}"
    );
    assert!(!text.contains(VISION_START_TOKEN), "{text}");

    // No `--mmproj`, no check: a text-only command never reads the vocabulary.
    assert!(VisionRequest::default()
        .load_service(
            "qwen35",
            &ModelVocabulary::from_tokens(&[]),
            cli_image_policy()
        )
        .expect("no projector requested")
        .is_none());
    let _ = std::fs::remove_dir_all(&dir);
}

/// The architecture check still comes first: a vocabulary cannot make a
/// non-Bonsai-2 model eligible, and the refusal says so.
#[test]
fn the_architecture_is_refused_before_the_vocabulary_is_read() {
    let dir = scratch("projector_arch_first");
    let mmproj = synthetic_projector(&dir);
    let err = with_projector(&mmproj)
        .load_service(
            "qwen3",
            &ModelVocabulary::from_tokens(&[]),
            cli_image_policy(),
        )
        .map(|_| ())
        .expect_err("a qwen3 model");
    let _ = std::fs::remove_dir_all(&dir);
    assert!(err.downcast_ref::<VisionMarkerError>().is_none());
    assert!(err.to_string().contains("'qwen3'"), "{err}");
}

// ── The commands hand their own model's vocabulary to the projector load ────

/// `run`, `chat` and `serve` each give the projector load the vocabulary of
/// the model they were pointed at: the test kit's synthetic hybrid carries no
/// `tokenizer.ggml.tokens`, so with `--mmproj` every one refuses at start-up
/// with the typed vocabulary refusal — before any weight is bound (and
/// `serve` before it binds a port).
#[test]
fn run_chat_and_serve_check_the_vocabulary_of_the_model_they_were_given() {
    use crate::cli::args::Cli;
    use crate::cli::util::test_env::{self, EnvVarGuard};
    use clap::Parser;

    let _env_lock = test_env::lock();
    let _no_model = EnvVarGuard::remove("OXI_MODEL");
    let dir = scratch("flow_vocabulary");
    let model = dir.join("synthetic-qwen35.gguf");
    std::fs::write(
        &model,
        oxibonsai_testkit::qwen35_fixture::synthetic_qwen35_gguf(),
    )
    .expect("write the synthetic hybrid");
    let model = model.to_string_lossy().into_owned();
    let mmproj = synthetic_projector(&dir);

    let dispatch = |args: &[&str]| -> String {
        let cli = Cli::try_parse_from(std::iter::once("oxibonsai").chain(args.iter().copied()))
            .expect("argv parses");
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        match runtime.block_on(crate::cli::run_with(cli)) {
            Ok(()) => "OK".to_string(),
            Err(e) => e.to_string(),
        }
    };
    let commands = [vec!["run", "-p", "hi"], vec!["chat"]]
        .into_iter()
        .chain(cfg!(feature = "server").then(|| vec!["serve"]));
    for command in commands {
        let mut args = command.clone();
        // A context inside the fixture's own 4096-token limit, so the
        // context guard that runs first has nothing to say.
        args.extend([
            "--model",
            model.as_str(),
            "--mmproj",
            mmproj.as_str(),
            "--ctx",
            "256",
        ]);
        let msg = dispatch(&args);
        assert!(
            msg.starts_with("[vision_vocabulary_mismatch] --mmproj "),
            "{command:?}: {msg}"
        );
        assert!(
            msg.contains("carries no vocabulary (tokenizer.ggml.tokens)"),
            "{command:?}: {msg}"
        );
        for (token, id) in BONSAI2_MARKERS {
            assert!(
                msg.contains(token) && msg.contains(&id.to_string()),
                "{command:?}: {msg}"
            );
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

// ── The real 27B's vocabulary (header only) ─────────────────────────────────

/// Real 27B (`OXI_BONSAI2_PQ2_GGUF`, and `OXI_BONSAI2_MMPROJ_GGUF` for the
/// projector): the language model's own `tokenizer.ggml.tokens` holds
/// `<|vision_start|>`, `<|vision_end|>` and `<|image_pad|>` at 248053,
/// 248054 and 248056 — the ids the splice layer uses — and the projector
/// loads against that vocabulary through [`VisionRequest::load_service`], the
/// path every real `--mmproj` command takes. Only the file's header is read
/// (no weight is bound), under the host's real-model lock like every other
/// real-27B test; self-skipping when the files are not named.
#[test]
fn real_27b_vocabulary_carries_the_image_tokens_at_the_splice_ids() {
    use oxibonsai_testkit::capability::{record_executed_timed, record_skipped, Capability};
    const TEST: &str =
        "oxibonsai-cli::bin::real_27b_vocabulary_carries_the_image_tokens_at_the_splice_ids";
    let Some(language_model) = crate::cli::test_fixtures::env_path(
        "OXI_BONSAI2_PQ2_GGUF",
        "the real Ternary-Bonsai-2-27B-PQ2_0.gguf",
    ) else {
        record_skipped(Capability::Bonsai2Models, TEST);
        return;
    };
    let _real = crate::cli::test_fixtures::real_model_lock();
    let started = std::time::Instant::now();

    let mmap = oxibonsai_core::gguf::reader::mmap_gguf_file(&language_model)
        .expect("map the real language model");
    let gguf = GgufFile::parse(&mmap).expect("parse the real language model's header");
    let arch = gguf
        .metadata
        .get_string(oxibonsai_core::gguf::tensor_info::keys::GENERAL_ARCHITECTURE)
        .expect("general.architecture");
    assert_eq!(arch, "qwen35");
    let vocabulary = ModelVocabulary::of_gguf(&gguf);
    assert_eq!(
        vocabulary.token_count(),
        248_320,
        "the real vocabulary is read from the file"
    );
    check_vision_markers(
        &language_model.to_string_lossy(),
        VisionTokenIds::BONSAI2,
        &vocabulary,
    )
    .expect("the real vocabulary holds the image tokens at the splice's ids");

    // The projector, loaded the way `run`/`chat`/`serve` load it.
    if let Some(mmproj) = crate::cli::test_fixtures::env_path(
        "OXI_BONSAI2_MMPROJ_GGUF",
        "the real Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf",
    ) {
        let request = VisionRequest {
            mmproj: Some(mmproj.to_string_lossy().into_owned()),
            ..VisionRequest::default()
        };
        let service = request
            .load_service(arch, &vocabulary, cli_image_policy())
            .expect("the real projector loads against the real vocabulary")
            .expect("a projector was requested");
        assert_eq!(service.token_ids(), VisionTokenIds::BONSAI2);
    }

    record_executed_timed(Capability::Bonsai2Models, TEST, started.elapsed());
}
