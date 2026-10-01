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
    run(Some(model.clone()), false).expect("text info");
    run(Some(model), true).expect("json info");
    let _ = std::fs::remove_file(&path);
}
