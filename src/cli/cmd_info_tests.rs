//! Tests of `oxibonsai info`: model resolution, the `--json` null sentinel,
//! and a full run over the synthetic `qwen35` hybrid in both output forms —
//! the path that resolves the hybrid's `--backend auto` executor (the Metal
//! hybrid runner on a Metal host) without building an engine.

use super::*;

use crate::cli::util::test_env::{lock, EnvVarGuard};

fn scratch_gguf(tag: &str, bytes: &[u8]) -> std::path::PathBuf {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let path = std::env::temp_dir().join(format!(
        "oxibonsai_cmd_info_{tag}_{}_{stamp}.gguf",
        std::process::id()
    ));
    std::fs::write(&path, bytes).expect("write the fixture");
    path
}

#[test]
fn an_explicit_model_wins_and_no_model_at_all_is_an_error() {
    let _env = lock();
    let _unset = EnvVarGuard::remove("OXI_MODEL");
    assert_eq!(
        resolve_model_path(Some("m.gguf".to_string())).expect("explicit"),
        "m.gguf"
    );
    let err = resolve_model_path(None).expect_err("no --model and no OXI_MODEL");
    assert!(err.to_string().contains("no model"), "{err}");
    let _empty = EnvVarGuard::set("OXI_MODEL", "");
    assert!(
        resolve_model_path(None).is_err(),
        "an empty OXI_MODEL is no model"
    );
    let _set = EnvVarGuard::set("OXI_MODEL", "env.gguf");
    assert_eq!(resolve_model_path(None).expect("env"), "env.gguf");
}

#[test]
fn the_dash_sentinel_becomes_json_null() {
    assert_eq!(null_if_dash("-"), None);
    assert_eq!(null_if_dash("42"), Some("42"));
}

fn request(model: &str, json: bool) -> InfoRequest {
    InfoRequest {
        model: Some(model.to_string()),
        json,
        ..InfoRequest::default()
    }
}

/// `info` over the synthetic hybrid succeeds in both forms: the report
/// resolves the `--backend auto` executor (and, on a Metal host, the Metal
/// runner's window and footprint) from a dry bind alone.
#[test]
fn info_reports_the_hybrid_in_text_and_json() {
    let path = scratch_gguf(
        "hybrid",
        &oxibonsai_testkit::qwen35_fixture::synthetic_qwen35_gguf(),
    );
    let model = path.to_string_lossy().into_owned();
    run(request(&model, false)).expect("text info");
    run(request(&model, true)).expect("json info");
    let _ = std::fs::remove_file(&path);
}

/// The synthetic projector widened to the synthetic hybrid's rows.
fn projector_bytes() -> Vec<u8> {
    let spec = oxibonsai_testkit::mmproj_fixture::MmprojFixtureSpec {
        projection_dim: oxibonsai_testkit::qwen35_fixture::HIDDEN,
        ..oxibonsai_testkit::mmproj_fixture::MmprojFixtureSpec::tiny()
    };
    oxibonsai_testkit::mmproj_fixture::synthetic_mmproj_gguf(&spec).expect("the projector")
}

/// `info --mmproj --prefill-chunk` over the synthetic hybrid succeeds in both
/// forms, and the report it prints is the one `run` acts on: under the load
/// options the projector and the chunk make (`VisionRequest::
/// hybrid_load_options`), the planned executor's window counts the Metal
/// tower and its prefill calls take the requested chunk — read off the plan,
/// whichever executor this host resolves `auto` to.
#[test]
fn info_with_a_projector_and_a_prefill_chunk_reports_the_plan_run_acts_on() {
    use oxibonsai_runtime::engine_hybrid_gpu::{HybridBackendPlan, HybridLoadScope};
    let model_path = scratch_gguf(
        "hybrid_vision",
        &oxibonsai_testkit::qwen35_fixture::synthetic_qwen35_gguf(),
    );
    let projector_path = scratch_gguf("projector", &projector_bytes());
    let model = model_path.to_string_lossy().into_owned();
    let projector = projector_path.to_string_lossy().into_owned();
    let vision = bonsai2::VisionRequest {
        mmproj: Some(projector.clone()),
        images: Vec::new(),
        image_max_tokens: Some(128),
    };
    for json in [false, true] {
        run(InfoRequest {
            vision: vision.clone(),
            prefill_chunk: Some(96),
            ..request(&model, json)
        })
        .expect("info with a projector and a chunk");
    }

    let options = vision
        .hybrid_load_options("qwen35", Some(96))
        .expect("the projector's options");
    let bytes = std::fs::read(&model_path).expect("read the model");
    let gguf = GgufFile::parse(&bytes).expect("parses");
    let report = {
        let _scope = HybridLoadScope::enter(options);
        model_desc::hybrid_report(&gguf).expect("the hybrid report")
    };
    let plan = report.prefill_chunk_plan().expect("the fixture binds");
    assert_eq!(plan.requested, Some(96));
    assert_eq!(plan.model_chunk, 96);
    assert_eq!(plan.in_effect, 96, "the fixture's window leaves room");
    assert!(!plan.capped());
    let lines = report.lines(bytes.len() as u64).join("\n");
    assert!(
        lines.contains("Prefill chunk: 96 tokens per prefill call (--prefill-chunk 96)"),
        "{lines}"
    );
    let json = report.to_json(bytes.len() as u64);
    assert_eq!(json["prefill_chunk"]["in_effect"], 96, "{json}");
    assert_eq!(json["prefill_chunk"]["requested"], 96, "{json}");
    match &report.backend_plan {
        Some(HybridBackendPlan::Metal { window, .. }) => {
            assert_eq!(window.vision_resident_bytes, options.vision_resident_bytes);
            assert!(
                options.vision_resident_bytes > 0,
                "a Metal tower has a footprint"
            );
            assert_eq!(
                json["backend"]["vision_resident_bytes"],
                options.vision_resident_bytes
            );
        }
        Some(HybridBackendPlan::Cpu { .. }) => {}
        None => panic!("the fixture binds"),
    }
    // Both towers' footprints come from the header alone.
    let footprints = projector_footprints(&projector, 128).expect("footprints");
    assert!(footprints.cpu_bytes > 0);
    assert_eq!(
        footprints.metal_bytes.is_some(),
        cfg!(all(feature = "metal", target_os = "macos")),
        "a Metal tower exactly on a Metal build"
    );
    let text = vision_lines(&footprints).join("\n");
    assert!(text.contains("CPU tower"), "{text}");
    assert!(
        text.contains("128-token images") || text.contains("no Metal tower"),
        "{text}"
    );
    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_file(&projector_path);
}

/// `info --mmproj` over a dense model is refused exactly as `run --mmproj`
/// refuses it: typed, naming both architectures.
#[test]
fn info_with_a_projector_over_a_dense_model_is_refused_as_not_a_hybrid_model() {
    let dense = scratch_gguf(
        "dense",
        &crate::cli::test_fixtures::tiny_dense_gguf(Vec::new()),
    );
    let projector = scratch_gguf("dense_projector", &projector_bytes());
    let err = run(InfoRequest {
        vision: bonsai2::VisionRequest {
            mmproj: Some(projector.to_string_lossy().into_owned()),
            images: Vec::new(),
            image_max_tokens: None,
        },
        ..request(&dense.to_string_lossy(), false)
    })
    .expect_err("a dense model has no rows prefill")
    .to_string();
    assert!(err.starts_with("[NOT_A_HYBRID_MODEL] --mmproj "), "{err}");
    let _ = std::fs::remove_file(&dense);
    let _ = std::fs::remove_file(&projector);
}

/// `info` over a vision projector itself (`clip`) succeeds in both forms and
/// reports what each executor's tower keeps resident, read from the header.
#[test]
fn info_reports_a_projectors_towers() {
    let path = scratch_gguf("clip", &projector_bytes());
    let model = path.to_string_lossy().into_owned();
    run(request(&model, false)).expect("text info");
    run(request(&model, true)).expect("json info");
    let _ = std::fs::remove_file(&path);
}

/// The prefill-chunk line names the request beside a capped call size, and
/// the model's default when there was no request.
#[test]
fn the_prefill_chunk_line_names_a_cap_and_a_default() {
    use model_desc::PrefillChunkPlan;
    let capped = PrefillChunkPlan {
        requested: Some(4096),
        model_chunk: 4096,
        in_effect: 1536,
    };
    assert!(capped.capped());
    let line = capped.describe();
    assert!(
        line.starts_with("1536 tokens per prefill call (--prefill-chunk 4096)"),
        "{line}"
    );
    assert!(line.contains("KV window"), "{line}");
    let default = PrefillChunkPlan {
        requested: None,
        model_chunk: 512,
        in_effect: 512,
    };
    assert_eq!(
        default.describe(),
        "512 tokens per prefill call (the model's default; --prefill-chunk changes it)"
    );
}
