//! Unit tests for `bonsai2.rs` (sibling file, declared there via `#[path]`,
//! so `super` still names that module).

use super::*;
use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue};

const GIB: u64 = 1024 * 1024 * 1024;

#[test]
fn qwen35_is_recognized_as_the_bonsai2_hybrid_family() {
    assert!(is_qwen35_hybrid("qwen35"));
    assert!(!is_qwen35_hybrid("qwen3"));
    assert!(!is_qwen35_hybrid("llama"));
    assert!(!is_qwen35_hybrid(""));
}

#[test]
fn default_max_seq_len_is_8192_for_qwen35_and_4096_otherwise() {
    assert_eq!(default_max_seq_len("qwen35"), 8192);
    assert_eq!(default_max_seq_len("qwen3"), 4096);
    assert_eq!(default_max_seq_len("clip"), 4096);
}

// ── A metadata-only qwen35 GGUF with the real 27B geometry ──────────────────

fn qwen35_27b_gguf_bytes(extra: Vec<(&str, MetadataWriteValue)>) -> Vec<u8> {
    let mut w = GgufWriter::new();
    crate::cli::test_fixtures::qwen35_27b_metadata(&mut w);
    for (key, value) in extra {
        w.add_metadata(key, value);
    }
    w.to_bytes().expect("serialize qwen35 metadata fixture")
}

fn qwen35_27b_config() -> HybridConfig {
    let bytes = qwen35_27b_gguf_bytes(Vec::new());
    let gguf = GgufFile::parse(&bytes).expect("parse fixture");
    HybridConfig::from_metadata(&gguf.metadata).expect("the 27B geometry is valid")
}

#[test]
fn hybrid_geometry_matches_the_real_27b_numbers_and_the_runtime_constants() {
    let geometry = HybridStateGeometry::from_config(&qwen35_27b_config());
    assert_eq!(geometry.n_layers, 64);
    assert_eq!(geometry.n_full_layers, 16);
    assert_eq!(geometry.n_linear_layers, 48);
    assert_eq!(geometry.kv_bytes_per_token, 65_536, "64 KiB/token at f16");
    assert_eq!(geometry.recurrent_bytes, 156_893_184);
    // The runtime's documented constants are the same numbers (the runtime
    // crate cross-checks the recurrent one against the real
    // `RecurrentCache::memory_bytes()`).
    assert_eq!(
        geometry.kv_bytes_per_token,
        oxibonsai_runtime::config::BONSAI2_KV_BYTES_PER_TOKEN
    );
    assert_eq!(
        geometry.recurrent_bytes,
        oxibonsai_runtime::config::BONSAI2_RECURRENT_BYTES
    );
    assert_eq!(
        geometry.kv_bytes_at(8192),
        512 * 1024 * 1024,
        "8192 ctx = 512 MiB KV"
    );
}

// ── check_context_budget (design §5.6) ──────────────────────────────────────

/// The 24 GiB M3 + real PQ2_0 file worked example of Appendix A.3.
fn pq2_on_24_gib(requested: usize) -> ContextBudgetInputs {
    ContextBudgetInputs {
        requested,
        model_context_length: 262_144,
        total_ram_bytes: 24 * GIB,
        weight_bytes: 7_206_168_928,
        recurrent_bytes: 156_893_184,
        kv_bytes_per_token: 65_536,
    }
}

#[test]
fn the_shipped_8192_default_is_accepted_with_the_appendix_a3_ram_limit() {
    let budget = check_context_budget(&pq2_on_24_gib(8192)).expect("8192 fits");
    assert_eq!(
        budget.ram_limit, 178_176,
        "Appendix A.3's corrected worked example"
    );
    assert_eq!(budget.effective_limit, 178_176);
}

#[test]
fn ctx_300000_is_refused_naming_both_limits_and_the_gib() {
    let msg = check_context_budget(&pq2_on_24_gib(300_000)).expect_err("300000 must be refused");
    assert!(msg.contains("300000"), "names the request: {msg}");
    assert!(msg.contains("262144"), "names the model limit: {msg}");
    assert!(msg.contains("178176"), "names the RAM-derived limit: {msg}");
    // 300000 x 64 KiB = 19_660_800_000 B = 18.31 GiB of KV cache.
    assert!(
        msg.contains("18.31 GiB"),
        "names the resulting KV GiB: {msg}"
    );
    // 24 GiB - 7.21 GB - 0.15 GB - 0.25 GiB - 6 GiB = 10.89 GiB usable.
    assert!(msg.contains("10.89 GiB"), "names the usable budget: {msg}");
    assert!(msg.contains("24.00 GiB"), "names the host RAM: {msg}");
}

#[test]
fn a_request_between_the_ram_limit_and_the_model_limit_is_refused_too() {
    // 262144 is within the model's own limit but over this host's RAM bound.
    let msg = check_context_budget(&pq2_on_24_gib(262_144)).expect_err("RAM-bound refusal");
    assert!(msg.contains("178176"), "{msg}");
    assert!(
        msg.contains("16.00 GiB"),
        "262144 x 64 KiB = 16 GiB of KV: {msg}"
    );
}

#[test]
fn unknown_host_ram_only_applies_the_model_limit() {
    let mut inputs = pq2_on_24_gib(262_144);
    inputs.total_ram_bytes = u64::MAX;
    check_context_budget(&inputs).expect("the model limit itself is allowed");
    inputs.requested = 300_000;
    let msg = check_context_budget(&inputs).expect_err("still over the model limit");
    assert!(msg.contains("262144"), "{msg}");
    assert!(msg.contains("unbounded"), "{msg}");
}

#[test]
fn non_qwen35_architecture_is_always_a_noop() {
    let bytes = qwen35_27b_gguf_bytes(Vec::new());
    let gguf = GgufFile::parse(&bytes).expect("parse fixture");
    // Even an absurd request is not this guard's business for a dense arch.
    validate_context_for_model(&gguf, "qwen3", 7_206_168_928, 10_000_000)
        .expect("non-qwen35 architectures are never guarded");
}

#[test]
fn qwen35_default_context_is_accepted_on_a_tiny_fixture() {
    // A tiny weight figure (the fixture's own size, far below the real 27B)
    // leaves the RAM-derived ceiling far above the shipped 8192 default on
    // any host: the model's own 262144 is the only binding limit.
    let bytes = qwen35_27b_gguf_bytes(Vec::new());
    let gguf = GgufFile::parse(&bytes).expect("parse fixture");
    validate_context_for_model(&gguf, "qwen35", bytes.len() as u64, 8192)
        .expect("the shipped 8192 default must be accepted");
}

#[test]
fn qwen35_ctx_300000_is_refused_naming_the_model_limit() {
    let bytes = qwen35_27b_gguf_bytes(Vec::new());
    let gguf = GgufFile::parse(&bytes).expect("parse fixture");
    // Refused on ANY host: 300000 exceeds the model's own 262144, whatever
    // the RAM-derived limit is.
    let err = validate_context_for_model(&gguf, "qwen35", bytes.len() as u64, 300_000)
        .expect_err("300000 exceeds the model's own 262144");
    let msg = err.to_string();
    assert!(msg.contains("300000"), "names the request: {msg}");
    assert!(msg.contains("262144"), "names the model's own limit: {msg}");
    assert!(
        msg.contains("18.31 GiB"),
        "names the KV GiB the request needs: {msg}"
    );
}

// ── `--think` default derived from the template ─────────────────────────────

#[test]
fn prompt_opens_think_block_tracks_the_last_open_and_close() {
    assert!(prompt_opens_think_block("<|im_start|>assistant\n<think>\n"));
    assert!(!prompt_opens_think_block(
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ));
    assert!(!prompt_opens_think_block("<|im_start|>assistant\n"));
    // An earlier, closed think block (a re-rendered assistant turn) does not
    // count; the final open one does.
    assert!(prompt_opens_think_block(
        "<think>\nold\n</think>\n\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n"
    ));
}

use crate::cli::test_fixtures::THINKING_TEMPLATE;

fn template_from_text(text: &str) -> ResolvedChatTemplate {
    let mut w = GgufWriter::new();
    w.add_metadata(
        "general.architecture",
        MetadataWriteValue::Str("qwen35".to_string()),
    );
    w.add_metadata(
        "tokenizer.chat_template",
        MetadataWriteValue::Str(text.to_string()),
    );
    let bytes = w.to_bytes().expect("serialize");
    let gguf = GgufFile::parse(&bytes).expect("parse");
    ResolvedChatTemplate::from_gguf(&gguf.metadata).expect("the test template compiles")
}

#[test]
fn default_enable_thinking_is_true_for_a_thinking_template() {
    let template = template_from_text(THINKING_TEMPLATE);
    assert!(matches!(template, ResolvedChatTemplate::Jinja(_)));
    assert!(default_enable_thinking(&template).expect("render probe"));
}

#[test]
fn default_enable_thinking_is_false_for_the_chatml_fallback() {
    let fallback = ResolvedChatTemplate::default_fallback();
    assert!(!default_enable_thinking(&fallback).expect("render probe"));
}

#[test]
fn explicit_no_think_closes_the_block_the_default_leaves_open() {
    let template = template_from_text(THINKING_TEMPLATE);
    let messages = [RenderMessage::new("user", "hi")];
    let off = template
        .render_with(
            &messages,
            &RenderOptions {
                add_generation_prompt: true,
                enable_thinking: Some(false),
                ..RenderOptions::default()
            },
        )
        .expect("render");
    assert!(!prompt_opens_think_block(&off));
}

// ── §5.7 vision flags: validate, then typed NOT_YET_SUPPORTED ───────────────

fn scratch(tag: &str) -> std::path::PathBuf {
    crate::cli::test_fixtures::scratch_dir(&format!("bonsai2_{tag}"))
}

#[test]
fn no_vision_flags_is_a_noop() {
    VisionRequest::default()
        .reject_until_supported()
        .expect("nothing to refuse");
}

#[test]
fn an_image_url_is_validated_then_refused_with_the_typed_code() {
    let request = VisionRequest {
        images: vec!["https://example.com/cat.png".to_string()],
        ..VisionRequest::default()
    };
    let msg = request
        .reject_until_supported()
        .expect_err("vision is not supported yet")
        .to_string();
    assert!(msg.starts_with("[NOT_YET_SUPPORTED]"), "{msg}");
    assert!(msg.contains("vision tower"), "{msg}");
    assert!(
        msg.contains("1024"),
        "names the default image budget: {msg}"
    );
}

#[test]
fn a_missing_image_file_is_a_validation_error_not_the_typed_refusal() {
    let dir = scratch("missing_image");
    let missing = dir.join("nope.png");
    let request = VisionRequest {
        images: vec![missing.to_string_lossy().into_owned()],
        ..VisionRequest::default()
    };
    let msg = request
        .reject_until_supported()
        .expect_err("missing file")
        .to_string();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(!msg.contains(NOT_YET_SUPPORTED), "{msg}");
    assert!(msg.contains("--image"), "{msg}");
}

#[test]
fn an_explicit_image_max_tokens_alone_is_refused_with_the_typed_code() {
    let request = VisionRequest {
        image_max_tokens: Some(512),
        ..VisionRequest::default()
    };
    let msg = request
        .reject_until_supported()
        .expect_err("refused")
        .to_string();
    assert!(msg.starts_with("[NOT_YET_SUPPORTED]"), "{msg}");
    assert!(msg.contains("512"), "{msg}");
}

#[test]
fn mmproj_must_be_a_clip_gguf() {
    let dir = scratch("mmproj");
    let language_model = dir.join("language.gguf");
    std::fs::write(&language_model, qwen35_27b_gguf_bytes(Vec::new())).expect("write");
    let request = VisionRequest {
        mmproj: Some(language_model.to_string_lossy().into_owned()),
        ..VisionRequest::default()
    };
    let msg = request
        .reject_until_supported()
        .expect_err("not a projector")
        .to_string();
    assert!(msg.contains("expected 'clip'"), "{msg}");

    let projector = dir.join("mmproj.gguf");
    let mut w = GgufWriter::new();
    w.add_metadata(
        "general.architecture",
        MetadataWriteValue::Str("clip".to_string()),
    );
    std::fs::write(&projector, w.to_bytes().expect("serialize")).expect("write");
    let request = VisionRequest {
        mmproj: Some(projector.to_string_lossy().into_owned()),
        ..VisionRequest::default()
    };
    let msg = request
        .reject_until_supported()
        .expect_err("valid projector, still not supported")
        .to_string();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(msg.starts_with("[NOT_YET_SUPPORTED]"), "{msg}");
}

#[test]
fn gib_renders_two_decimals() {
    assert_eq!(gib(GIB), "1.00 GiB");
    assert_eq!(gib(19_660_800_000), "18.31 GiB");
}
