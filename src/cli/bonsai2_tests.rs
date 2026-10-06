//! Unit tests for `bonsai2.rs` (sibling file, declared there via `#[path]`,
//! so `super` still names that module).

use super::*;
use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue};
use oxibonsai_core::MetadataValue;

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
        .validate(true, &no_sources())
        .expect("nothing to check for run/chat");
    request
        .validate(false, &no_sources())
        .expect("nothing to check for serve");
    assert!(request
        .load_service("qwen35", &bonsai2_vocabulary(), cli_image_policy())
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
        .validate(true, &no_sources())
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
        .validate(true, &no_sources())
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
            .validate(true, &no_sources())
            .expect_err("out of range")
            .to_string();
        assert!(msg.contains("1..=16384"), "{bad}: {msg}");
    }
    for good in [1usize, 1024, 16_384] {
        with_projector(&mmproj, &[], Some(good))
            .validate(true, &no_sources())
            .expect("in range");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_remote_image_url_is_refused_with_its_typed_code() {
    let _env_lock = test_env::lock();
    let _unset = EnvVarGuard::remove(ALLOW_IMAGE_URL_FETCH_ENV);
    let dir = scratch("remote");
    let mmproj = metadata_only_projector(&dir);
    let url = "https://example.com/cat.png";
    let msg = with_projector(&mmproj, &[url], None)
        .validate(true, &no_sources())
        .expect_err("not opted in")
        .to_string();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(msg.starts_with("[image_url_fetch_disabled]"), "{msg}");
    assert!(msg.contains("--image https://example.com/cat.png"), "{msg}");
    assert!(msg.contains("server-side request forgery"), "{msg}");

    // An opt-in with no fetcher installed (what a library caller's policy
    // holds when it sets the opt-in alone) is refused too, saying why.
    let opted_in = cli_image_policy()
        .with_remote_access(oxibonsai_model::vision::RemoteImageAccess::OptedInWithoutFetcher);
    let msg = validate_image_ref(url, &opted_in)
        .expect_err("nothing to fetch it with")
        .to_string();
    assert!(msg.starts_with("[image_url_fetch_disabled]"), "{msg}");
    assert!(msg.contains("installed no fetcher"), "{msg}");
}

#[test]
fn a_missing_image_file_is_refused_naming_the_flag() {
    let dir = scratch("missing_image");
    let mmproj = metadata_only_projector(&dir);
    let missing = dir.join("nope.png").to_string_lossy().into_owned();
    let msg = with_projector(&mmproj, &[missing.as_str()], None)
        .validate(true, &no_sources())
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
        .validate(true, &no_sources())
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
        .validate(true, &no_sources())
        .expect_err("not a projector")
        .to_string();
    assert!(msg.contains("expected 'clip'"), "{msg}");

    let projector = metadata_only_projector(&dir);
    with_projector(&projector, &[], None)
        .validate(true, &no_sources())
        .expect("a clip GGUF passes the flag validation");
    // ...but a projector without the tower's tensors is refused at load,
    // with the file named.
    let msg = with_projector(&projector, &[], None)
        .load_service("qwen35", &bonsai2_vocabulary(), cli_image_policy())
        .expect_err("no tower in the file")
        .to_string();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(msg.contains("--mmproj"), "{msg}");
}

#[test]
fn serve_refuses_the_image_flag() {
    // A server's validation also checks the media directory, which the
    // environment can name: hold the lock and clear it, so a sibling test's
    // `OXI_MEDIA_PATH` cannot reach this one.
    let _env_lock = crate::cli::util::test_env::lock();
    #[cfg(feature = "server")]
    let _no_media_env = crate::cli::util::test_env::EnvVarGuard::remove(MEDIA_PATH_ENV);
    let dir = scratch("serve_image");
    let mmproj = metadata_only_projector(&dir);
    let png = pattern_png_file(&dir);
    let msg = with_projector(&mmproj, &[png.as_str()], None)
        .validate(false, &no_sources())
        .expect_err("a server takes images from requests")
        .to_string();
    with_projector(&mmproj, &[], None)
        .validate(false, &no_sources())
        .expect("--mmproj alone is how serve enables vision");
    let _ = std::fs::remove_dir_all(&dir);
    assert!(msg.contains("image_url"), "{msg}");
}

#[test]
fn the_projector_serves_only_a_qwen35_language_model() {
    let dir = scratch("projector_arch");
    let mmproj = synthetic_projector(&dir);
    let msg = with_projector(&mmproj, &[], None)
        .load_service("qwen3", &bonsai2_vocabulary(), cli_image_policy())
        .expect_err("not a Bonsai 2 model")
        .to_string();
    assert!(msg.contains("`qwen35`"), "{msg}");
    assert!(msg.contains("'qwen3'"), "{msg}");
    // The engine's own code, so the CLI and the server name it alike.
    assert!(msg.starts_with("[NOT_A_HYBRID_MODEL] --mmproj "), "{msg}");
    // The same refusal sizes no engine: it comes before the model is loaded.
    let early = with_projector(&mmproj, &[], None)
        .hybrid_load_options("qwen3", Some(64))
        .expect_err("not a Bonsai 2 model")
        .to_string();
    assert_eq!(early, msg, "one refusal, wherever it is raised");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The image tokens are checked before any weight is bound, against the ids
/// every projector load splices with: nothing to check without `--mmproj`, a
/// vocabulary that agrees passes, and one that lacks them is the typed
/// refusal naming every token.
#[test]
fn the_image_tokens_are_checked_before_the_model_is_loaded() {
    let dir = scratch("image_tokens");
    let mmproj = metadata_only_projector(&dir);
    let lacking = vocabulary_tokens(600, &[]);
    VisionRequest::default()
        .check_image_tokens(&ModelVocabulary::from_tokens(&lacking))
        .expect("no projector, nothing to check");
    let request = with_projector(&mmproj, &[], None);
    request
        .check_image_tokens(&bonsai2_vocabulary())
        .expect("the Bonsai 2 vocabulary agrees");
    let err = request
        .check_image_tokens(&ModelVocabulary::from_tokens(&lacking))
        .expect_err("no image tokens");
    assert_eq!(err.problems.len(), 3, "{err}");
    assert!(
        err.to_string()
            .starts_with("[vision_vocabulary_mismatch] --mmproj "),
        "{err}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The projector's header sizes a hybrid engine's load options (its Metal
/// tower's resident bytes, the prefill chunk) before any language-model
/// weight is bound; the tower is then loaded for the executor the engine
/// decodes on and checked against the engine — a projector whose rows are
/// not as wide as the model's is refused, typed, before any image is
/// encoded, and a dense engine is refused for its model kind.
#[test]
fn the_projector_sizes_the_engine_and_is_loaded_for_its_executor() {
    use oxibonsai_runtime::engine_hybrid_gpu::HybridBackend;
    use oxibonsai_runtime::vision_prefill::VisionService;
    use oxibonsai_testkit::mmproj_fixture::{synthetic_mmproj_gguf, MmprojFixtureSpec};
    use oxibonsai_testkit::qwen35_fixture::{synthetic_qwen35_gguf, HIDDEN};

    let dir = scratch("load_for_engine");
    let fits = dir.join("mmproj_fits.gguf");
    let spec = MmprojFixtureSpec {
        projection_dim: HIDDEN,
        ..MmprojFixtureSpec::tiny()
    };
    std::fs::write(&fits, synthetic_mmproj_gguf(&spec).expect("projector")).expect("write");
    let fits_path = fits.to_string_lossy().into_owned();
    // The test kit's tiny projector emits 80-wide rows; the model reads
    // `HIDDEN`-wide ones.
    let narrow = synthetic_projector(&dir);

    let request = with_projector(&fits_path, &[], Some(256));
    let options = request
        .hybrid_load_options("qwen35", Some(1024))
        .expect("load options");
    assert_eq!(options.prefill_chunk, Some(1024));
    assert_eq!(
        options.vision_resident_bytes,
        VisionService::metal_footprint(&fits, 256).expect("footprint")
    );
    assert_eq!(
        VisionRequest::default()
            .hybrid_load_options("qwen35", None)
            .expect("no projector"),
        oxibonsai_runtime::engine_hybrid_gpu::HybridLoadOptions::default(),
        "without --mmproj the engine is built exactly as before"
    );
    // No projector and a dense model: nothing to check, nothing to refuse.
    assert!(VisionRequest::default()
        .hybrid_load_options("qwen3", Some(8))
        .is_ok());

    let model = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&model).expect("fixture parses");
    for backend in [
        oxibonsai_runtime::engine_seam::Backend::Cpu,
        oxibonsai_runtime::engine_seam::Backend::Auto,
    ] {
        let engine = {
            let _scope = oxibonsai_runtime::engine_hybrid_gpu::HybridLoadScope::enter(options);
            oxibonsai_runtime::InferenceEngine::from_gguf_with_backend(
                &gguf,
                oxibonsai_runtime::sampling::SamplingParams::default(),
                7,
                64,
                backend,
            )
            .expect("a hybrid engine")
        };
        let service = request
            .load_service_for(
                "qwen35",
                &bonsai2_vocabulary(),
                || Ok(cli_image_policy()),
                &engine,
            )
            .expect("a matching projector loads")
            .expect("a projector was requested");
        assert_eq!(
            service.tower().is_metal(),
            engine.hybrid_backend() == Some(HybridBackend::Metal),
            "{backend}: the tower of the engine's own executor"
        );
        if let Some(window) = engine.hybrid_metal_window() {
            assert_eq!(
                window.vision_resident_bytes,
                service.tower().resident_bytes() as u64,
                "{backend}: the window left room for exactly the tower that was built"
            );
        }
        let err = with_projector(&narrow, &[], None)
            .load_service_for(
                "qwen35",
                &bonsai2_vocabulary(),
                || Ok(cli_image_policy()),
                &engine,
            )
            .expect_err("rows of another width")
            .to_string();
        assert!(err.contains("[projector_mismatch]"), "{backend}: {err}");
    }

    // A dense engine is refused for its model kind, typed.
    let dense = oxibonsai_runtime::InferenceEngine::new(
        oxibonsai_core::config::Qwen3Config::tiny_test(),
        oxibonsai_runtime::sampling::SamplingParams::default(),
        7,
    );
    let err = request
        .load_service_for(
            "qwen35",
            &bonsai2_vocabulary(),
            || Ok(cli_image_policy()),
            &dense,
        )
        .expect_err("a dense engine serves no image turn");
    let runtime = err
        .downcast_ref::<oxibonsai_runtime::error::RuntimeError>()
        .expect("the engine's typed refusal");
    assert_eq!(
        oxibonsai_runtime::engine_seam::engine_error_code(runtime),
        Some("NOT_A_HYBRID_MODEL")
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A command without `--mmproj` never builds an image policy: the loader
/// answers "no projector" before it would call the policy, so the
/// remote-image settings (which only the policy reads) cannot stop a
/// text-only `run` or `chat`. With a projector the policy is built, and its
/// error is the loader's.
#[test]
fn the_image_policy_is_built_only_when_a_projector_is_loaded() {
    use std::cell::Cell;
    let dense = oxibonsai_runtime::InferenceEngine::new(
        oxibonsai_core::config::Qwen3Config::tiny_test(),
        oxibonsai_runtime::sampling::SamplingParams::default(),
        7,
    );
    let built = Cell::new(0usize);
    let refusing_policy = || -> anyhow::Result<oxibonsai_model::vision::ImageSourcePolicy> {
        built.set(built.get() + 1);
        anyhow::bail!("OXI_IMAGE_URL_TIMEOUT_MS=\"abc\" is not a number of milliseconds")
    };
    for arch in ["qwen3", "qwen35"] {
        let service = VisionRequest::default()
            .load_service_for(arch, &bonsai2_vocabulary(), refusing_policy, &dense)
            .expect("no projector: nothing to load, and no policy to build");
        assert!(service.is_none(), "{arch}: no projector was requested");
    }
    assert_eq!(
        built.get(),
        0,
        "a text-only command must never build the image policy"
    );

    // With a projector the policy is built, and a refusal of its stops the
    // load before any tower is built.
    let dir = std::env::temp_dir().join(format!(
        "oxibonsai-lazy-image-policy-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let projector = synthetic_projector(&dir);
    let err = with_projector(&projector, &[], None)
        .load_service_for("qwen35", &bonsai2_vocabulary(), refusing_policy, &dense)
        .expect_err("the policy's own refusal")
        .to_string();
    assert!(err.contains("OXI_IMAGE_URL_TIMEOUT_MS"), "{err}");
    assert_eq!(built.get(), 1, "a projector load builds the policy once");
    let _ = std::fs::remove_dir_all(&dir);
}

// ── The image tokens against the model's vocabulary ─────────────────────────

/// A vocabulary of `len` tokens, every one empty but the `markers`
/// (`(token, id)` pairs): an empty token costs no allocation, so a list as
/// long as the real 248,320-token one is cheap.
pub(crate) fn vocabulary_tokens(len: usize, markers: &[(&str, u32)]) -> Vec<MetadataValue> {
    let mut tokens = vec![MetadataValue::String(String::new()); len];
    for &(token, id) in markers {
        tokens[id as usize] = MetadataValue::String(token.to_string());
    }
    tokens
}

/// The three image tokens at the ids the Bonsai 2 vocabulary gives them
/// (design Appendix A.1).
pub(crate) const BONSAI2_MARKERS: [(&str, u32); 3] = [
    (VISION_START_TOKEN, 248_053),
    (VISION_END_TOKEN, 248_054),
    (IMAGE_PAD_TOKEN, 248_056),
];

/// A vocabulary that holds the three image tokens where the splice layer
/// expects them — what every test that loads a projector for a Bonsai 2 model
/// passes as the model's vocabulary.
pub(crate) fn bonsai2_vocabulary() -> ModelVocabulary<'static> {
    static TOKENS: std::sync::OnceLock<Vec<MetadataValue>> = std::sync::OnceLock::new();
    ModelVocabulary::from_tokens(
        TOKENS.get_or_init(|| vocabulary_tokens(248_057, &BONSAI2_MARKERS)),
    )
}

/// A metadata-only `qwen35` GGUF (the real 27B's geometry) whose
/// `tokenizer.ggml.tokens` is `tokens`; `None` leaves the array out.
pub(crate) fn qwen35_gguf_with_vocabulary(tokens: Option<&[MetadataValue]>) -> Vec<u8> {
    let extra = tokens
        .map(|tokens| {
            let strings = tokens
                .iter()
                .map(|token| token.as_str().unwrap_or_default().to_string())
                .collect();
            vec![(
                "tokenizer.ggml.tokens",
                MetadataWriteValue::ArrayStr(strings),
            )]
        })
        .unwrap_or_default();
    qwen35_27b_gguf_bytes(extra)
}

#[test]
fn a_synthetic_projector_prepares_and_encodes_a_png_file_and_a_data_uri() {
    let dir = scratch("projector_encode");
    let mmproj = synthetic_projector(&dir);
    let png = pattern_png_file(&dir);
    let request = with_projector(&mmproj, &[png.as_str()], None);
    request.validate(true, &no_sources()).expect("valid flags");
    let service = request
        .load_service("qwen35", &bonsai2_vocabulary(), cli_image_policy())
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
        .validate(true, &no_sources())
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
        .load_service("qwen35", &bonsai2_vocabulary(), cli_image_policy())
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

/// CRC-32 (IEEE), for sealing the chunks of a hand-built PNG.
fn crc32(parts: &[&[u8]]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for byte in parts.iter().flat_map(|part| part.iter()) {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                0xEDB8_8320 ^ (crc >> 1)
            } else {
                crc >> 1
            };
        }
    }
    crc ^ 0xFFFF_FFFF
}

/// A PNG file that is only a header and an end marker, declaring a
/// `width` x `height` 8-bit RGB image: a few dozen bytes that would cost
/// gigabytes to decode if they carried the pixels they promise.
fn header_only_png_file(dir: &std::path::Path, width: u32, height: u32) -> String {
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    let mut png = vec![0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n'];
    png.extend_from_slice(&13u32.to_be_bytes());
    png.extend_from_slice(b"IHDR");
    png.extend_from_slice(&ihdr);
    png.extend_from_slice(&crc32(&[b"IHDR", &ihdr]).to_be_bytes());
    png.extend_from_slice(&0u32.to_be_bytes());
    png.extend_from_slice(b"IEND");
    png.extend_from_slice(&crc32(&[b"IEND"]).to_be_bytes());
    let path = dir.join(format!("header_only_{width}x{height}.png"));
    std::fs::write(&path, png).expect("write the header-only PNG");
    path.to_string_lossy().into_owned()
}

/// `--image-max-tokens` also bounds what a source image may cost to decode:
/// the policy a loaded projector resolves references under carries the
/// decode budget that follows from it, and an image declaring more pixels is
/// `image_too_large` from its header — not a gigabyte of inflate and unfilter
/// that ends as a decode error. A larger token budget admits a larger source.
#[test]
fn a_source_image_past_the_decode_budget_is_refused_from_its_header() {
    use oxibonsai_model::vision::preprocess::source_pixel_budget;
    let dir = scratch("projector_decode_budget");
    let mmproj = synthetic_projector(&dir);
    // 8192 x 8192: at the hard decode limit, far past the default budget.
    let huge = header_only_png_file(&dir, 8192, 8192);
    // 4032 x 3024: a 12-megapixel phone photo, inside the default budget.
    let photo = header_only_png_file(&dir, 4032, 3024);

    let request = with_projector(&mmproj, &[huge.as_str()], None);
    request.validate(true, &no_sources()).expect("valid flags");
    let service = request
        .load_service("qwen35", &bonsai2_vocabulary(), cli_image_policy())
        .expect("loads")
        .expect("requested");
    assert_eq!(
        service.policy().max_source_pixels,
        Some(source_pixel_budget(1024, 32)),
        "the default 1024-token budget"
    );
    let msg = request
        .prepare_images(&service)
        .expect_err("far past the decode budget")
        .to_string();
    assert!(msg.starts_with("[image_too_large]"), "{msg}");
    assert!(msg.contains("--image-max-tokens"), "{msg}");

    // The photo gets past the gate and fails only on its missing pixels.
    let request = with_projector(&mmproj, &[photo.as_str()], None);
    let msg = request
        .prepare_images(&service)
        .expect_err("no image data")
        .to_string();
    assert!(msg.starts_with("[image_decode_failed]"), "{msg}");

    // A budget of 16384 tokens lifts the limit to the hard cap, which the
    // 8192 x 8192 header meets exactly: past the gate again.
    let request = with_projector(&mmproj, &[huge.as_str()], Some(16_384));
    let service = request
        .load_service("qwen35", &bonsai2_vocabulary(), cli_image_policy())
        .expect("loads")
        .expect("requested");
    assert_eq!(
        service.policy().max_source_pixels,
        Some(oxibonsai_model::vision::image_decode::MAX_DECODED_PIXELS)
    );
    let msg = request
        .prepare_images(&service)
        .expect_err("no image data")
        .to_string();
    assert!(msg.starts_with("[image_decode_failed]"), "{msg}");

    // A policy that already carries a budget keeps it.
    let mut explicit = cli_image_policy();
    explicit.max_source_pixels = Some(1_000);
    let service = request
        .load_service("qwen35", &bonsai2_vocabulary(), explicit)
        .expect("loads")
        .expect("requested");
    assert_eq!(service.policy().max_source_pixels, Some(1_000));
    let _ = std::fs::remove_dir_all(&dir);
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

// ── --allow-image-url-fetch / --media-path ──────────────────────────────────

use crate::cli::args::{Cli, Commands};
use crate::cli::util::test_env::{self, EnvVarGuard};
use clap::Parser;
#[cfg(feature = "server")]
use oxibonsai_model::vision::{load_image_bytes, ImageSourcePolicy};

/// No image-source flag given.
fn no_sources() -> ImageSourceFlags {
    ImageSourceFlags::default()
}

fn parse_cli(args: &[&str]) -> Result<Cli, clap::Error> {
    Cli::try_parse_from(std::iter::once("oxibonsai").chain(args.iter().copied()))
}

/// `--allow-image-url-fetch` is a flag of `run`, `chat` and `serve`: present
/// it is `true`, absent `false`.
#[test]
fn the_url_fetch_opt_in_flag_parses_on_every_command_that_takes_it() {
    let allow_of = |args: &[&str]| -> bool {
        match parse_cli(args).expect("parses").command {
            Commands::Run {
                allow_image_url_fetch,
                ..
            }
            | Commands::Chat {
                allow_image_url_fetch,
                ..
            } => allow_image_url_fetch,
            #[cfg(feature = "server")]
            Commands::Serve {
                allow_image_url_fetch,
                ..
            } => allow_image_url_fetch,
            _ => panic!("not a vision command: {args:?}"),
        }
    };
    assert!(allow_of(&[
        "run",
        "-p",
        "hi",
        "--mmproj",
        "m.gguf",
        "--allow-image-url-fetch"
    ]));
    assert!(!allow_of(&["run", "-p", "hi", "--mmproj", "m.gguf"]));
    assert!(allow_of(&[
        "chat",
        "--mmproj",
        "m.gguf",
        "--allow-image-url-fetch"
    ]));
    assert!(!allow_of(&["chat", "--mmproj", "m.gguf"]));
    #[cfg(feature = "server")]
    {
        assert!(allow_of(&[
            "serve",
            "--mmproj",
            "m.gguf",
            "--allow-image-url-fetch"
        ]));
        assert!(!allow_of(&["serve", "--mmproj", "m.gguf"]));
    }
}

/// A hybrid `qwen35` model embeds (on the CPU model): neither the flag's help
/// nor its `model` value's own text may still say it has no embedder and
/// answers `501`.
#[cfg(feature = "server")]
#[test]
fn the_embedding_backend_help_does_not_claim_a_hybrid_model_has_no_embedder() {
    use clap::CommandFactory;
    let command = Cli::command();
    let serve = command.find_subcommand("serve").expect("serve subcommand");
    let flag = serve
        .get_arguments()
        .find(|arg| arg.get_id() == "embedding_backend")
        .expect("the --embedding-backend flag");
    let help = flag
        .get_long_help()
        .or_else(|| flag.get_help())
        .map(ToString::to_string)
        .unwrap_or_default();
    let model_value_help = flag
        .get_possible_values()
        .into_iter()
        .find(|value| value.get_name() == "model")
        .and_then(|value| value.get_help().map(ToString::to_string))
        .unwrap_or_default();
    for text in [&help, &model_value_help] {
        assert!(!text.contains("no embedder"), "{text}");
        assert!(!text.contains("has none yet"), "{text}");
        assert!(!text.contains("honest 501"), "{text}");
    }
    assert!(
        help.contains("hybrid"),
        "the flag says hybrids embed: {help}"
    );
    assert!(help.contains("CPU"), "and on what: {help}");
    assert!(model_value_help.contains("hybrid"), "{model_value_help}");
}

/// `--media-path` belongs to `serve` alone: it parses there and is an
/// unknown argument on `run` and `chat`.
#[cfg(feature = "server")]
#[test]
fn the_media_path_flag_parses_on_serve_only() {
    let cli = parse_cli(&["serve", "--mmproj", "m.gguf", "--media-path", "media"]).expect("parses");
    match cli.command {
        Commands::Serve { media_path, .. } => assert_eq!(media_path.as_deref(), Some("media")),
        _ => panic!("expected Serve"),
    }
    match parse_cli(&["serve"]).expect("bare serve").command {
        Commands::Serve { media_path, .. } => assert_eq!(media_path, None),
        _ => panic!("expected Serve"),
    }
    for args in [
        vec!["run", "-p", "hi", "--media-path", "media"],
        vec!["chat", "--media-path", "media"],
    ] {
        let err = parse_cli(&args)
            .err()
            .unwrap_or_else(|| panic!("{args:?} must not parse"));
        assert!(
            err.to_string().contains("--media-path"),
            "the parser names the unknown flag: {err}"
        );
    }
}

#[test]
fn the_url_fetch_opt_in_is_the_flag_else_the_environment() {
    for (flag, env, expected) in [
        (false, None, false),
        (true, None, true),
        (false, Some("1"), true),
        (false, Some("true"), true),
        (false, Some(" YES "), true),
        (false, Some("On"), true),
        (false, Some("0"), false),
        (false, Some("false"), false),
        (false, Some(""), false),
        (false, Some("enable"), false),
        // A flag is an explicit opt-in: no environment value takes it back.
        (true, Some("0"), true),
        (true, Some("1"), true),
    ] {
        assert_eq!(
            resolve_url_fetch_opt_in(flag, env),
            expected,
            "flag {flag}, env {env:?}"
        );
    }
}

#[cfg(feature = "server")]
#[test]
fn the_media_directory_is_the_flag_else_the_environment_else_none() {
    let root = |flag: Option<&str>, env: Option<&str>| resolve_media_root(flag, env);
    assert_eq!(root(None, None), None);
    assert_eq!(
        root(None, Some("  ")),
        None,
        "a blank environment value is none"
    );
    let from_env = root(None, Some(" /srv/media ")).expect("the environment names one");
    assert_eq!(from_env.path, std::path::PathBuf::from("/srv/media"));
    assert_eq!(from_env.source, MediaRootSource::Env);
    let from_flag = root(Some("flag-dir"), Some("/srv/media")).expect("the flag names one");
    assert_eq!(
        from_flag.path,
        std::path::PathBuf::from("flag-dir"),
        "the flag wins over the environment"
    );
    assert_eq!(from_flag.source, MediaRootSource::Flag);
    assert_eq!(MediaRootSource::Flag.label(), "--media-path");
    assert_eq!(MediaRootSource::Env.label(), "OXI_MEDIA_PATH");
}

/// The policy `run` / `chat` resolve under: the flag turns the opt-in on
/// whatever the environment says, and with no flag the environment decides.
#[test]
fn the_cli_policy_follows_the_flag_then_the_environment() {
    let _env_lock = test_env::lock();
    let flagged = ImageSourceFlags {
        allow_image_url_fetch: true,
        media_path: None,
        ..ImageSourceFlags::default()
    };
    // An opted-in policy carries the fetcher; one without the opt-in none.
    let fetches = |flags: &ImageSourceFlags| {
        let policy = flags.cli_policy().expect("policy");
        assert_eq!(
            policy.remote.is_opted_in(),
            policy.remote.fetcher().is_some(),
            "an opt-in always comes with the fetcher"
        );
        policy.remote.is_opted_in()
    };
    {
        let _unset = EnvVarGuard::remove(ALLOW_IMAGE_URL_FETCH_ENV);
        assert!(!fetches(&no_sources()));
        assert!(fetches(&flagged), "the flag alone");
        assert!(!cli_image_policy().remote.is_opted_in());
    }
    {
        let _on = EnvVarGuard::set(ALLOW_IMAGE_URL_FETCH_ENV, "1");
        assert!(fetches(&no_sources()), "the environment alone");
        assert!(fetches(&flagged));
        assert!(cli_image_policy().remote.is_opted_in());
    }
    {
        let _off = EnvVarGuard::set(ALLOW_IMAGE_URL_FETCH_ENV, "0");
        assert!(!fetches(&no_sources()));
        assert!(fetches(&flagged), "the flag beats a `0`");
    }
    // The CLI names its own files whatever the opt-in says.
    let policy = flagged.cli_policy().expect("policy");
    assert!(policy.allow_any_local_path && policy.media_root.is_none());
}

/// Without the opt-in a remote `--image` is refused, naming the opt-in.
/// With it — the flag or the environment — the fetcher is installed, and
/// validation checks the URL without fetching it: a public name passes (it
/// is fetched once, when the image is prepared), a loopback literal is
/// refused by the address policy before any connection.
#[test]
fn the_url_fetch_opt_in_installs_the_fetcher_and_validation_never_fetches() {
    let _env_lock = test_env::lock();
    let _unset = EnvVarGuard::remove(ALLOW_IMAGE_URL_FETCH_ENV);
    let url = "https://example.com/cat.png";
    let check = |flags: &ImageSourceFlags, url: &str| {
        validate_image_ref(url, &flags.cli_policy().expect("policy")).map_err(|e| e.to_string())
    };
    let without = check(&no_sources(), url).expect_err("not opted in");
    assert!(
        without.starts_with("[image_url_fetch_disabled]"),
        "{without}"
    );
    assert!(without.contains("--allow-image-url-fetch"), "{without}");
    assert!(without.contains("OXI_ALLOW_IMAGE_URL_FETCH=1"), "{without}");
    assert!(without.contains("server-side request forgery"), "{without}");

    let flagged = ImageSourceFlags {
        allow_image_url_fetch: true,
        ..ImageSourceFlags::default()
    };
    check(&flagged, url).expect("a public URL passes validation, unfetched");
    let loopback = check(&flagged, "http://127.0.0.1:9/cat.png").expect_err("loopback");
    assert!(loopback.starts_with("[image_url_refused]"), "{loopback}");
    assert!(loopback.contains("not a public address"), "{loopback}");
    let credentials = check(&flagged, "https://user:pw@example.com/cat.png").expect_err("userinfo");
    assert!(
        credentials.starts_with("[image_url_refused]"),
        "{credentials}"
    );

    // The environment opts in the same way.
    let _on = EnvVarGuard::set(ALLOW_IMAGE_URL_FETCH_ENV, "1");
    check(&no_sources(), url).expect("opted in by the environment");
}

/// Flags that would silently do nothing are refused, naming the flag — the
/// same rule `--image-max-tokens` follows.
#[test]
fn the_new_flags_without_a_projector_are_refused_as_silent_no_ops() {
    for (sources, flag) in [
        (
            ImageSourceFlags {
                allow_image_url_fetch: true,
                media_path: None,
                ..ImageSourceFlags::default()
            },
            "--allow-image-url-fetch",
        ),
        (
            ImageSourceFlags {
                allow_image_url_fetch: false,
                media_path: Some("media".to_string()),
                ..ImageSourceFlags::default()
            },
            "--media-path",
        ),
        (
            ImageSourceFlags {
                image_url_timeout_ms: Some(500),
                ..ImageSourceFlags::default()
            },
            "--image-url-timeout-ms",
        ),
        (
            ImageSourceFlags {
                image_url_allow_hosts: vec!["images.intranet".to_string()],
                ..ImageSourceFlags::default()
            },
            "--image-url-allow-host",
        ),
    ] {
        let msg = VisionRequest::default()
            .validate(false, &sources)
            .expect_err("no projector")
            .to_string();
        assert!(msg.contains(&format!("{flag} has no effect")), "{msg}");
        assert!(msg.contains("--mmproj"), "{msg}");
    }
    // With a projector both are accepted (the directory must exist, below).
    let dir = scratch("flags_with_projector");
    let mmproj = metadata_only_projector(&dir);
    let media = dir.join("media");
    std::fs::create_dir_all(&media).expect("media dir");
    let sources = ImageSourceFlags {
        allow_image_url_fetch: true,
        media_path: Some(media.to_string_lossy().into_owned()),
        ..ImageSourceFlags::default()
    };
    with_projector(&mmproj, &[], None)
        .validate(false, &sources)
        .expect("a projector makes both meaningful");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn media_path_is_for_serve_not_for_run_or_chat() {
    let dir = scratch("media_for_run");
    let mmproj = metadata_only_projector(&dir);
    let sources = ImageSourceFlags {
        allow_image_url_fetch: false,
        media_path: Some(dir.to_string_lossy().into_owned()),
        ..ImageSourceFlags::default()
    };
    let msg = with_projector(&mmproj, &[], None)
        .validate(true, &sources)
        .expect_err("run and chat read the files --image names")
        .to_string();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(msg.contains("--media-path is for `serve`"), "{msg}");
}

/// The media directory is checked before any model is loaded: it must exist
/// and be a directory, and the message names the setting that named it.
#[cfg(feature = "server")]
#[test]
fn a_media_directory_that_is_not_one_is_refused_at_startup_naming_the_setting() {
    let _env_lock = test_env::lock();
    let dir = scratch("media_startup");
    let mmproj = metadata_only_projector(&dir);
    let file = dir.join("not_a_directory.txt");
    std::fs::write(&file, b"x").expect("write");
    let missing = dir.join("missing");
    let request = with_projector(&mmproj, &[], None);
    let check = |media_path: Option<String>| {
        request
            .validate(
                false,
                &ImageSourceFlags {
                    allow_image_url_fetch: false,
                    media_path,
                    ..ImageSourceFlags::default()
                },
            )
            .map_err(|e| e.to_string())
    };

    let _unset = EnvVarGuard::remove(MEDIA_PATH_ENV);
    let msg = check(Some(missing.to_string_lossy().into_owned())).expect_err("missing");
    assert!(msg.contains("--media-path"), "{msg}");
    assert!(msg.contains("cannot be resolved"), "{msg}");
    let msg = check(Some(file.to_string_lossy().into_owned())).expect_err("a file");
    assert!(
        msg.contains("--media-path") && msg.contains("not a directory"),
        "{msg}"
    );
    let msg = check(Some(String::new())).expect_err("empty");
    assert!(msg.contains("--media-path is empty"), "{msg}");
    check(Some(dir.to_string_lossy().into_owned())).expect("a real directory");
    check(None).expect("no media directory is fine: data URIs only");

    // The environment fallback is held to the same standard, by its own name.
    let _bad = EnvVarGuard::set(MEDIA_PATH_ENV, &missing);
    let msg = check(None).expect_err("the environment names a missing directory");
    assert!(msg.contains("OXI_MEDIA_PATH"), "{msg}");
    assert!(!msg.contains("--media-path"), "{msg}");
    // ...and a flag that is given takes the environment out of the picture.
    check(Some(dir.to_string_lossy().into_owned())).expect("the flag wins");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A tiny PNG file written into `dir`, for the `file://` tests.
#[cfg(feature = "server")]
fn media_fixture(dir: &std::path::Path, name: &str) {
    let bytes = load_image_bytes(PATTERN_PNG_DATA_URI, &ImageSourcePolicy::local_user())
        .expect("the embedded PNG decodes from base64");
    std::fs::write(dir.join(name), bytes).expect("write the fixture");
}

/// `--media-path`: `file://` references resolve inside that directory and
/// nowhere else — through the policy the flag builds.
#[cfg(feature = "server")]
#[test]
fn the_media_path_flag_confines_file_references_to_its_directory() {
    let _env_lock = test_env::lock();
    let _unset = EnvVarGuard::remove(MEDIA_PATH_ENV);
    let base = scratch("media_confine");
    let media = base.join("media");
    std::fs::create_dir_all(media.join("nested")).expect("media dir");
    media_fixture(&media, "pic.png");
    media_fixture(&media.join("nested"), "deep.png");
    media_fixture(&base, "secret.png");
    let policy = ImageSourceFlags {
        allow_image_url_fetch: false,
        media_path: Some(media.to_string_lossy().into_owned()),
        ..ImageSourceFlags::default()
    }
    .server_policy()
    .expect("the directory resolves");
    assert!(
        !policy.allow_any_local_path,
        "a server never reads arbitrary paths"
    );
    assert_eq!(
        policy.media_root,
        Some(media.canonicalize().expect("canonical")),
        "the root is canonicalised once, at startup"
    );

    for inside in [
        "file://pic.png",
        "file://nested/deep.png",
        "file://./nested/../pic.png",
    ] {
        let result = load_image_bytes(inside, &policy);
        if inside.contains("..") {
            assert_eq!(
                result.expect_err(inside).code(),
                "image_file_refused",
                "{inside}"
            );
        } else {
            assert!(!result.expect(inside).is_empty(), "{inside}");
        }
    }
    let absolute_outside = format!("file://{}", base.join("secret.png").display());
    for escape in [
        "file://../secret.png",
        "file://nested/../../secret.png",
        absolute_outside.as_str(),
    ] {
        let err = load_image_bytes(escape, &policy).expect_err(escape);
        assert_eq!(err.code(), "image_file_refused", "{escape}: {err}");
    }
    // A plain path is not a `file://` reference: a server never reads it.
    let plain = base.join("secret.png").to_string_lossy().into_owned();
    let err = load_image_bytes(&plain, &policy).expect_err("plain path");
    assert_eq!(err.code(), "image_file_refused", "{err}");
    let _ = std::fs::remove_dir_all(&base);
}

/// The symlink case, through the flag's policy: a link inside the media
/// directory that leaves it is refused with the typed error.
#[cfg(all(feature = "server", unix))]
#[test]
fn a_symlink_out_of_the_media_path_is_refused_with_a_typed_error() {
    let _env_lock = test_env::lock();
    let _unset = EnvVarGuard::remove(MEDIA_PATH_ENV);
    let base = scratch("media_symlink");
    let media = base.join("media");
    let outside = base.join("outside");
    std::fs::create_dir_all(&media).expect("media dir");
    std::fs::create_dir_all(&outside).expect("outside dir");
    media_fixture(&outside, "secret.png");
    std::os::unix::fs::symlink(outside.join("secret.png"), media.join("link.png"))
        .expect("file link");
    std::os::unix::fs::symlink(&outside, media.join("dir_link")).expect("dir link");
    let policy = ImageSourceFlags {
        allow_image_url_fetch: false,
        media_path: Some(media.to_string_lossy().into_owned()),
        ..ImageSourceFlags::default()
    }
    .server_policy()
    .expect("the directory resolves");
    for escape in ["file://link.png", "file://dir_link/secret.png"] {
        let err = load_image_bytes(escape, &policy).expect_err(escape);
        assert_eq!(err.code(), "image_file_refused", "{escape}: {err}");
        assert!(
            err.to_string().contains("outside the media directory"),
            "{err}"
        );
    }
    let _ = std::fs::remove_dir_all(&base);
}

/// Flag > environment > default for the media directory, through the
/// policy the command builds.
#[cfg(feature = "server")]
#[test]
fn the_server_policy_follows_the_flag_then_the_environment_then_none() {
    let _env_lock = test_env::lock();
    let base = scratch("media_precedence");
    let flag_dir = base.join("from_flag");
    let env_dir = base.join("from_env");
    std::fs::create_dir_all(&flag_dir).expect("flag dir");
    std::fs::create_dir_all(&env_dir).expect("env dir");
    let canonical = |dir: &std::path::Path| dir.canonicalize().expect("canonical");
    let with_flag = ImageSourceFlags {
        allow_image_url_fetch: false,
        media_path: Some(flag_dir.to_string_lossy().into_owned()),
        ..ImageSourceFlags::default()
    };

    let _unset = EnvVarGuard::remove(MEDIA_PATH_ENV);
    assert_eq!(
        no_sources().server_policy().expect("policy").media_root,
        None
    );
    assert_eq!(
        with_flag.server_policy().expect("policy").media_root,
        Some(canonical(&flag_dir))
    );
    {
        let _env = EnvVarGuard::set(MEDIA_PATH_ENV, &env_dir);
        assert_eq!(
            no_sources().server_policy().expect("policy").media_root,
            Some(canonical(&env_dir)),
            "the environment is the fallback"
        );
        assert_eq!(
            with_flag.server_policy().expect("policy").media_root,
            Some(canonical(&flag_dir)),
            "the flag wins"
        );
    }
    // The opt-in reaches the server policy from either source.
    let _fetch = EnvVarGuard::remove(ALLOW_IMAGE_URL_FETCH_ENV);
    assert!(!no_sources()
        .server_policy()
        .expect("policy")
        .remote
        .is_opted_in());
    let opted_in = ImageSourceFlags {
        allow_image_url_fetch: true,
        media_path: None,
        ..ImageSourceFlags::default()
    };
    let served = opted_in.server_policy().expect("policy");
    assert!(
        served.remote.fetcher().is_some(),
        "the opt-in installs the fetcher on the server policy too"
    );
    let _ = std::fs::remove_dir_all(&base);
}

/// The two flags travel from the parsed command line into the command's
/// validation: `run`, `chat` and `serve` refuse them without `--mmproj`,
/// by name, before any model is opened (the model paths below do not exist).
#[test]
fn the_flags_reach_each_commands_validation() {
    let _env_lock = test_env::lock();
    let _no_model = EnvVarGuard::remove("OXI_MODEL");
    let dispatch = |args: &[&str]| -> String {
        let cli = parse_cli(args).expect("argv parses");
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        match runtime.block_on(crate::cli::run_with(cli)) {
            Ok(()) => "OK".to_string(),
            Err(e) => e.to_string(),
        }
    };
    let missing = std::env::temp_dir().join("oxibonsai-no-such-model.gguf");
    let missing = missing.to_string_lossy().into_owned();
    #[cfg_attr(not(feature = "server"), allow(unused_mut))]
    let mut commands = vec![vec!["run", "-p", "hi"], vec!["chat"]];
    #[cfg(feature = "server")]
    commands.push(vec!["serve"]);
    for command in &commands {
        let mut args = command.clone();
        args.extend(["--model", missing.as_str(), "--allow-image-url-fetch"]);
        let msg = dispatch(&args);
        assert!(
            msg.contains("--allow-image-url-fetch has no effect without --mmproj"),
            "{command:?}: {msg}"
        );
        for (flag, value) in [
            ("--image-url-timeout-ms", "500"),
            ("--image-url-allow-host", "images.intranet:8080"),
        ] {
            let mut args = command.clone();
            args.extend(["--model", missing.as_str(), flag, value]);
            let msg = dispatch(&args);
            assert!(
                msg.contains(&format!("{flag} has no effect without --mmproj")),
                "{command:?} {flag}: {msg}"
            );
        }
    }
    #[cfg(feature = "server")]
    {
        let msg = dispatch(&[
            "serve",
            "--model",
            missing.as_str(),
            "--allow-image-url-fetch",
        ]);
        assert!(
            msg.contains("--allow-image-url-fetch has no effect without --mmproj"),
            "{msg}"
        );
        let msg = dispatch(&[
            "serve",
            "--model",
            missing.as_str(),
            "--media-path",
            "media",
        ]);
        assert!(
            msg.contains("--media-path has no effect without --mmproj"),
            "{msg}"
        );
    }
}
