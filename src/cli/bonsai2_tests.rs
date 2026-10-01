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

// ── §5.7 vision flags: validation, projector loading, image preparation ────

fn scratch(tag: &str) -> std::path::PathBuf {
    crate::cli::test_fixtures::scratch_dir(&format!("bonsai2_{tag}"))
}

/// A `clip` GGUF with metadata only: it passes the flag validation (which
/// checks the architecture) but carries no tower.
fn metadata_only_projector(dir: &std::path::Path) -> String {
    let projector = dir.join("mmproj_metadata_only.gguf");
    let mut w = GgufWriter::new();
    w.add_metadata(
        "general.architecture",
        MetadataWriteValue::Str("clip".to_string()),
    );
    std::fs::write(&projector, w.to_bytes().expect("serialize")).expect("write");
    projector.to_string_lossy().into_owned()
}

/// The test kit's tiny synthetic Qwen3-VL projector (projection width 80),
/// written to `dir`.
pub(crate) fn synthetic_projector(dir: &std::path::Path) -> String {
    use oxibonsai_testkit::mmproj_fixture::{synthetic_mmproj_gguf, MmprojFixtureSpec};
    let bytes = synthetic_mmproj_gguf(&MmprojFixtureSpec::tiny()).expect("synthetic projector");
    let path = dir.join("mmproj_tiny.gguf");
    std::fs::write(&path, bytes).expect("write");
    path.to_string_lossy().into_owned()
}

/// A 256 x 192 RGB PNG (red = x, green = a vertical ramp, a blue
/// checkerboard of 32-pixel squares): the Bonsai 2 vision golden's geometry,
/// so a 6 x 8 merged grid at the default budget. Embedded because the
/// repository ignores `*.png` files.
pub(crate) const PATTERN_PNG_DATA_URI: &str =
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAQAAAADACAIAAABkjyoxAAAEY0lEQVR42u3TYasYAhiG4XOE\
kK1ZsxDLNBqhEWIhxEKjTZtM6NAmhNAmZEIj1ohlQugQi4XQaIQQQrNMJtNoNGKxEEKT/Yh1Pt1XvZ/frp66BwcGhgYHBkbyThrh\
/2/z82/HDQqAv+zfQQD8Zf+OAuAv+3cSAH/Zv7MA+Mv+XQTAX/bvKgD+sn83AfCX/bsLgL/sHyUA/rJ/tAD4y/4xAuAv+/cQAH/Z\
P1YA/GX/OAHwl/17CoC/7B8vAP6yfy8B8Jf9ewuAv+zfRwD8Zf++AuAv+/cTAH/ZP0EA/GX//gLgL/snCoC/7D9AAPxl/yQB8Jf9\
BwqAv+w/SAD8Zf9kAfCX/QcLgL/sP2TYQPxh/6EC4C/7DxMAf9l/uAD4y/4pAuAv+48QAH/Zf6QA+Mv+owTAX/YfLQD+sv8YAfCX\
/ccKgL/sP04A/GX/VAHwl/3HC4C/7D9BAPxl/4kC4E/7BcBf9p8sAP6y/xQB8Jf9pwqAv+w/TQD8Zf80AfCX/acLgL/sP0MA/GX/\
mQLgL/unC4C/7D9LAPxl/9kC4C/7ZwiAv+yfKQD+sv8cAfCX/bMEwF/2zx4yEH/Yf64A+Mv+8wTAX/bPEQB/2X++APjL/gsEwF/2\
XygA/rL/IgHwl/1DAuAv+y8WAH/Zf4kA+Mv+uQLgL/vnCYC/7L9UAPxl/2UC4C/7LxcAf9l/hQD4y/4rBcBf9l8lAP6y/2oB8Jf9\
1wiAv+y/VgD8Zf91AuAv++cLgL/sXyAA/rL/egHwl/03CIC/7L9RAPxl/00C4C/7bxYAf9m/UAD8Zf8tAuAv+28dNhB/2H+bAPjL\
/tsFwF/2LxIAf9l/hwD4y/47BcBf9t8lAP6y/24B8Jf9iwXAX/YvEQB/2X+PAPjL/nsFwF/23ycA/rL/fgHwl/1LBcBf9j8gAP6y\
f5kA+Mv+BwXAX/Y/JAD+sv9hAfCX/Y8IgL/sf1QA/GX/YwLgL/sfFwB/2T8sAP6y/wkB8Jf9TwqAv+x/SgD8Zf/TAuAv+5cLgL/s\
f0YA/GX/swLgL/tXDBmIP+x/TgD8Zf/zAuAv+18QAH/Z/6IA+Mv+lwTAX/a/LAD+sn+lAPjL/lcEwF/2vyoA/rJ/lQD4y/7XBMBf\
9r8uAP6y/w0B8Jf9bwqAv+x/SwD8ab8A+Mv+dwTAX/a/KwD+sv89AfCX/e8LgL/s/0AA/GX/hwLgL/s/EgB/2f+xAPjL/tUC4C/7\
PxEAf9m/RgD8Zf+nAuAv+9cKgL/s/0wA/GX/5wLgL/vXDRuIP+z/QgD8Zf+XAuAv+9cLgL/s/0oA/GX/BgHwl/1fC4C/7P9GAPxl\
/0YB8Jf93wqAv+z/TgD8Zf8mAfCX/d8LgL/s/0EA/GX/jwLgL/t/EgB/2b9ZAPxl/88C4C/7fxEAf9m/RQD8Zf+vAuAv+38TAH/Z\
/7sA+Mv+PwTAX/b/KQD+sv8vAfCX/X8LgL/s/0cA/GX/vwLgL/u3CoC/7P9PAPxl//8C4O/6twGhXx9+85z8hQAAAABJRU5ErkJggg==";

/// [`PATTERN_PNG_DATA_URI`] written to `dir` as a PNG file (the `--image
/// <path>` form).
pub(crate) fn pattern_png_file(dir: &std::path::Path) -> String {
    let bytes = oxibonsai_model::vision::load_image_bytes(
        PATTERN_PNG_DATA_URI,
        &oxibonsai_model::vision::ImageSourcePolicy::local_user(),
    )
    .expect("the embedded PNG decodes from base64");
    let path = dir.join("pattern_256x192.png");
    std::fs::write(&path, bytes).expect("write the test image");
    path.to_string_lossy().into_owned()
}

/// A 1 x 4000 8-bit grayscale PNG: too elongated for any small budget
/// without distorting its aspect ratio.
const STRIP_PNG_DATA_URI: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAA+gCAAAAAAedin3\
                                  AAAAH0lEQVR42u3CAQkAAAACoKY3vSGJpgEAAAAAAACAewNGt9Bqp3bDOgAAAABJRU5ErkJggg==";

fn with_projector(mmproj: &str, images: &[&str], budget: Option<usize>) -> VisionRequest {
    VisionRequest {
        mmproj: Some(mmproj.to_string()),
        images: images.iter().map(|s| s.to_string()).collect(),
        image_max_tokens: budget,
    }
}

#[test]
fn no_vision_flags_is_a_noop() {
    let request = VisionRequest::default();
    assert!(request.is_empty());
    request
        .validate(true)
        .expect("nothing to check for run/chat");
    request.validate(false).expect("nothing to check for serve");
    assert!(request
        .load_service("qwen35", cli_image_policy())
        .expect("no projector requested")
        .is_none());
    assert_eq!(request.effective_image_max_tokens(), 1024);
}

#[test]
fn an_image_without_mmproj_is_refused_naming_the_missing_flag() {
    let request = VisionRequest {
        images: vec!["cat.png".to_string()],
        ..VisionRequest::default()
    };
    let msg = request
        .validate(true)
        .expect_err("no projector")
        .to_string();
    assert!(msg.contains("--image needs the vision projector"), "{msg}");
    assert!(msg.contains("--mmproj"), "{msg}");
}

#[test]
fn an_image_max_tokens_without_mmproj_is_refused_as_a_silent_no_op() {
    let request = VisionRequest {
        image_max_tokens: Some(512),
        ..VisionRequest::default()
    };
    let msg = request
        .validate(true)
        .expect_err("no projector")
        .to_string();
    assert!(msg.contains("--image-max-tokens has no effect"), "{msg}");
    assert!(msg.contains("--mmproj"), "{msg}");
}

#[test]
fn the_image_budget_must_be_in_range() {
    let dir = scratch("budget");
    let mmproj = metadata_only_projector(&dir);
    for bad in [0usize, 16_385] {
        let msg = with_projector(&mmproj, &[], Some(bad))
            .validate(true)
            .expect_err("out of range")
            .to_string();
        assert!(msg.contains("1..=16384"), "{bad}: {msg}");
    }
    for good in [1usize, 1024, 16_384] {
        with_projector(&mmproj, &[], Some(good))
            .validate(true)
            .expect("in range");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_remote_image_url_is_refused_with_its_typed_code() {
    let dir = scratch("remote");
    let mmproj = metadata_only_projector(&dir);
    let url = "https://example.com/cat.png";
    let msg = with_projector(&mmproj, &[url], None)
        .validate(true)
        .expect_err("never fetched")
        .to_string();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(msg.starts_with("[image_url_fetch_disabled]"), "{msg}");
    assert!(msg.contains("--image https://example.com/cat.png"), "{msg}");
    assert!(msg.contains("server-side request forgery"), "{msg}");

    // The operator opt-in changes the reason, never the outcome.
    let mut opted_in = cli_image_policy();
    opted_in.allow_remote_fetch = true;
    let msg = validate_image_ref(url, &opted_in)
        .expect_err("still never fetched")
        .to_string();
    assert!(msg.starts_with("[image_url_fetch_disabled]"), "{msg}");
    assert!(msg.contains("no image fetcher"), "{msg}");
}

#[test]
fn a_missing_image_file_is_refused_naming_the_flag() {
    let dir = scratch("missing_image");
    let mmproj = metadata_only_projector(&dir);
    let missing = dir.join("nope.png").to_string_lossy().into_owned();
    let msg = with_projector(&mmproj, &[missing.as_str()], None)
        .validate(true)
        .expect_err("missing file")
        .to_string();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(msg.contains("--image"), "{msg}");
    assert!(msg.contains("cannot read"), "{msg}");
}

#[test]
fn an_invalid_data_uri_is_refused_with_its_typed_code() {
    let dir = scratch("data_uri");
    let mmproj = metadata_only_projector(&dir);
    let msg = with_projector(&mmproj, &["data:image/png;base64,@@@@"], None)
        .validate(true)
        .expect_err("not base64")
        .to_string();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(msg.starts_with("[image_data_uri_invalid]"), "{msg}");
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
        .validate(true)
        .expect_err("not a projector")
        .to_string();
    assert!(msg.contains("expected 'clip'"), "{msg}");

    let projector = metadata_only_projector(&dir);
    with_projector(&projector, &[], None)
        .validate(true)
        .expect("a clip GGUF passes the flag validation");
    // ...but a projector without the tower's tensors is refused at load,
    // with the file named.
    let msg = with_projector(&projector, &[], None)
        .load_service("qwen35", cli_image_policy())
        .expect_err("no tower in the file")
        .to_string();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(msg.contains("--mmproj"), "{msg}");
}

#[test]
fn serve_refuses_the_image_flag() {
    let dir = scratch("serve_image");
    let mmproj = metadata_only_projector(&dir);
    let png = pattern_png_file(&dir);
    let msg = with_projector(&mmproj, &[png.as_str()], None)
        .validate(false)
        .expect_err("a server takes images from requests")
        .to_string();
    with_projector(&mmproj, &[], None)
        .validate(false)
        .expect("--mmproj alone is how serve enables vision");
    let _ = std::fs::remove_dir_all(&dir);
    assert!(msg.contains("image_url"), "{msg}");
}

#[test]
fn the_projector_serves_only_a_qwen35_language_model() {
    let dir = scratch("projector_arch");
    let mmproj = synthetic_projector(&dir);
    let msg = with_projector(&mmproj, &[], None)
        .load_service("qwen3", cli_image_policy())
        .expect_err("not a Bonsai 2 model")
        .to_string();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(msg.contains("`qwen35`"), "{msg}");
    assert!(msg.contains("'qwen3'"), "{msg}");
}

#[test]
fn a_synthetic_projector_prepares_and_encodes_a_png_file_and_a_data_uri() {
    let dir = scratch("projector_encode");
    let mmproj = synthetic_projector(&dir);
    let png = pattern_png_file(&dir);
    let request = with_projector(&mmproj, &[png.as_str()], None);
    request.validate(true).expect("valid flags");
    let service = request
        .load_service("qwen35", cli_image_policy())
        .expect("the tiny projector loads")
        .expect("a projector was requested");
    assert_eq!(service.preprocess().max_tokens, 1024);

    let prepared = request.prepare_images(&service).expect("prepare");
    assert_eq!(prepared.len(), 1);
    assert_eq!(
        (prepared[0].grid.h, prepared[0].grid.w),
        (6, 8),
        "256 x 192 is already on the 32-pixel grid"
    );
    assert_eq!(prepared[0].source, (256, 192));

    let encoded = request
        .encode_prepared(&service, &prepared)
        .expect("encode");
    assert_eq!(encoded.len(), 1);
    assert_eq!(encoded[0].grid, prepared[0].grid);
    let projection = service.tower().config().projection_dim;
    assert_eq!(encoded[0].rows.len(), 48 * projection);
    assert!(encoded[0].rows.iter().all(|v| v.is_finite()));

    // The same pixels as a data URI encode to the same rows, bit for bit.
    let as_uri = with_projector(&mmproj, &[PATTERN_PNG_DATA_URI], None);
    as_uri
        .validate(true)
        .expect("a data URI is a valid --image");
    let from_uri = as_uri
        .encode_prepared(&service, &as_uri.prepare_images(&service).expect("prepare"))
        .expect("encode");
    let _ = std::fs::remove_dir_all(&dir);
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(from_uri[0].grid, encoded[0].grid);
    assert_eq!(bits(&from_uri[0].rows), bits(&encoded[0].rows));
}

#[test]
fn an_image_that_cannot_fit_the_budget_is_refused_with_its_typed_code() {
    let dir = scratch("projector_budget");
    let mmproj = synthetic_projector(&dir);
    let request = with_projector(&mmproj, &[STRIP_PNG_DATA_URI], Some(8));
    let service = request
        .load_service("qwen35", cli_image_policy())
        .expect("loads")
        .expect("requested");
    let msg = request
        .prepare_images(&service)
        .expect_err("a 1 x 4000 strip cannot keep its aspect in 8 tokens")
        .to_string();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(msg.starts_with("[image_too_many_tokens]"), "{msg}");
    assert!(msg.contains("--image data:image/png;base64,"), "{msg}");
    assert!(
        msg.len() < 1024,
        "a data URI is shortened in the message: {msg}"
    );
}

#[test]
fn prompt_rows_expands_each_placeholder_to_its_image_rows() {
    let ids = oxibonsai_model::vision::VisionTokenIds::BONSAI2;
    let grid = oxibonsai_model::vision::GridSize { h: 6, w: 8 };
    let text = [1u32, 2, 3];
    assert_eq!(prompt_rows(&text, &[grid], ids).expect("text only"), 3);
    let with_image = [1u32, ids.vision_start, ids.image_pad, ids.vision_end, 2];
    assert_eq!(
        prompt_rows(&with_image, &[grid], ids).expect("one image"),
        5 - 1 + 48
    );
    let msg = prompt_rows(&with_image, &[grid, grid], ids)
        .expect_err("one placeholder, two images")
        .to_string();
    assert!(
        msg.starts_with("[image_placeholder_count_mismatch]"),
        "{msg}"
    );
}

#[test]
fn gib_renders_two_decimals() {
    assert_eq!(gib(GIB), "1.00 GiB");
    assert_eq!(gib(19_660_800_000), "18.31 GiB");
}
