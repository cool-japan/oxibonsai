//! Unit tests for `cmd_validate.rs` (sibling file, declared there via
//! `#[path]`, so `super` still names that module).

use super::*;

/// A `qwen35` file whose metadata parses but whose tensors are missing: the
/// hybrid dry bind fails, so `validate` must NOT say OK (REQUIRED #2).
#[test]
fn a_qwen35_file_that_cannot_be_bound_is_not_ok() {
    use oxibonsai_core::gguf::writer::{GgufWriter, TensorEntry, TensorType};
    let mut w = GgufWriter::new();
    crate::cli::test_fixtures::qwen35_27b_metadata(&mut w);
    // The two tensors `validate`'s coverage check looks for — and nothing a
    // hybrid bind needs.
    for name in ["token_embd.weight", "output_norm.weight"] {
        w.add_tensor(TensorEntry {
            name: name.to_string(),
            shape: vec![8],
            tensor_type: TensorType::F32,
            data: vec![0u8; 32],
        });
    }
    let bytes = w.to_bytes().expect("serialize");
    let dir = crate::cli::test_fixtures::scratch_dir("validate_unbindable");
    let path = dir.join("Broken-qwen35.gguf");
    std::fs::write(&path, bytes).expect("write");
    let result = run(path.to_string_lossy().into_owned());
    let _ = std::fs::remove_dir_all(&dir);
    let err = result.expect_err("an unbindable hybrid must not validate");
    assert!(
        err.to_string().contains("cannot be bound"),
        "the refusal must name the bind failure: {err}"
    );
}

/// Whether `run` could build its engine for `path` — the load step of
/// `oxibonsai run` (on the CPU backend, a tiny context).
fn run_can_load(path: &std::path::Path) -> bool {
    let Ok(mmap) = oxibonsai_core::gguf::reader::mmap_gguf_file(path) else {
        return false;
    };
    let Ok(gguf) = GgufFile::parse(&mmap) else {
        return false;
    };
    let params = crate::cli::util::build_sampling_params(0.0, 40, 0.9, 1.0);
    let loads = oxibonsai_runtime::InferenceEngine::from_gguf_with_backend(
        &gguf,
        params,
        42,
        16,
        oxibonsai_runtime::engine_seam::Backend::Cpu,
    )
    .is_ok();
    loads
}

/// REQUIRED #2's acceptance test: `validate` and `run` agree on EVERY
/// `models/*.gguf` present (`OXIBONSAI_MODELS_DIR`) — OK exactly when the
/// engine loads. One model at a time (the real-model lock).
#[test]
fn validate_and_run_agree_on_every_real_model() {
    let Some(dir) = crate::cli::test_fixtures::env_path(
        "OXIBONSAI_MODELS_DIR",
        "the directory holding the real GGUFs",
    ) else {
        return;
    };
    let _real = crate::cli::test_fixtures::real_model_lock();
    let mut entries: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
        .expect("read models dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|ext| ext == "gguf"))
        .collect();
    entries.sort();
    assert!(!entries.is_empty(), "OXIBONSAI_MODELS_DIR holds no .gguf");
    let mut table = Vec::new();
    for path in &entries {
        let validate_ok = run(path.to_string_lossy().into_owned()).is_ok();
        let run_ok = run_can_load(path);
        table.push(format!(
            "{}: validate={} run={}",
            path.file_name()
                .map(|n| n.to_string_lossy())
                .unwrap_or_default(),
            if validate_ok { "OK" } else { "refused" },
            if run_ok { "loads" } else { "refused" }
        ));
        assert_eq!(
            validate_ok,
            run_ok,
            "validate and run disagree on {}",
            path.display()
        );
    }
    eprintln!("validate/run agreement:\n  {}", table.join("\n  "));
}
