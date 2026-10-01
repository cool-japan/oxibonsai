//! Image content through the OpenAI-compatible chat endpoints (SV-11;
//! bonsai2-design.md §6.2).
//!
//! # Synthetic cases (every run)
//!
//! A server over the test kit's synthetic `qwen35` hybrid (hidden width
//! 256, vocabulary 512) with the test kit's tiny Qwen3-VL projector widened
//! to the same 256 (its rows are the language model's embedding width), a
//! byte-level tokenizer carrying the chat and vision markers, and a compact
//! chat template that renders an image part exactly as the Bonsai 2
//! template does (`<|vision_start|><|image_pad|><|vision_end|>`):
//!
//! 1. an `image_url` data URI round-trips through `/v1/chat/completions`
//!    and `/v1/chat/completions/extended`: the answer is the engine's own
//!    multimodal generation of the same prompt (every per-token logprob
//!    equal), `usage.prompt_tokens` counts the image's 48 rows, and a
//!    different image changes the first step's distribution;
//! 2. text-only requests are byte-unchanged: the same answer with and
//!    without a projector loaded, for string and for text-part content, and
//!    on an engine that just served an image request (an image request
//!    after a text one, and two image requests in a row, likewise match a
//!    fresh engine);
//! 3. every refusal carries its named `error.code`: no projector, a
//!    malformed data URI, a remote URL, a file reference the server policy
//!    does not allow, an image over the token budget, too many images, and —
//!    on both endpoints — a prompt whose image rows overflow the context;
//! 4. streaming an image request, on either endpoint, delivers the
//!    non-streaming answer: the same text, the same finish reason and the
//!    same usage (the image's 48 rows included).
//!
//! # Real-model case
//!
//! `real_27b_server_round_trip_reproduces_the_golden_prompt_one`: the real
//! Bonsai 2 27B `PQ2_0` and its projector behind the same router, prompt 1
//! of the vision golden as a data-URI request (`enable_thinking = false`,
//! greedy, 32 tokens) on the CPU path — the golden text byte for byte, 67
//! prompt tokens (48 of them image rows, the fork server's own
//! `usage.prompt_tokens`) and 32 completion tokens. The models come from
//! `$OXI_BONSAI2_PQ2_GGUF` / `$OXI_BONSAI2_MMPROJ_GGUF`, else the release
//! file names under `$OXIBONSAI_MODELS_DIR` or the workspace `models/`
//! directory; absent files skip with a `bonsai2-vision` / `executed: false`
//! capability record (a failure under `OXI_REQUIRE_MODEL_FILES=1`).

#![cfg(feature = "server")]

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use tower::ServiceExt;

use oxibonsai_core::gguf::reader::{mmap_gguf_file, GgufFile};
use oxibonsai_model::vision::{ImageSourcePolicy, VisionTokenIds, VisionTower};
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::engine_pool::EnginePool;
use oxibonsai_runtime::engine_seam::Backend;
use oxibonsai_runtime::metrics::InferenceMetrics;
use oxibonsai_runtime::sampling::SamplingParams;
use oxibonsai_runtime::server::{create_router, create_router_full, RequestLimits, RouterOptions};
use oxibonsai_runtime::vision_prefill::{ChatPrompt, MultimodalPrompt, VisionService};
use oxibonsai_runtime::TokenizerBridge;
use oxibonsai_testkit::mmproj_fixture::{synthetic_mmproj_gguf, MmprojFixtureSpec};
use oxibonsai_testkit::qwen35_fixture::{synthetic_qwen35_gguf, CONTEXT_LENGTH, HIDDEN, VOCAB};
use oxibonsai_tokenizer::chat_templates::{
    RenderContentPart, RenderMessage, RenderOptions, ResolvedChatTemplate,
};
use oxibonsai_tokenizer::jinja::JinjaTemplate;

// ─────────────────────────────────────────────────────────────────────────────
// Shared fixtures
// ─────────────────────────────────────────────────────────────────────────────

/// The vendored Bonsai 2 vision golden (the model crate's fixture
/// directory: the fork's answers and the 256 x 192 image they describe).
fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../oxibonsai-model/tests/fixtures/bonsai2_golden_vision")
}

/// The golden's own image — read only by the real-model case, whose answer
/// is pinned to exactly these pixels.
fn golden_fixture_png() -> Vec<u8> {
    std::fs::read(golden_dir().join("fixture_256x192.png")).expect("the vendored fixture image")
}

/// Standard base64 with padding (RFC 4648 §4).
fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        for (i, shift) in [18u32, 12, 6, 0].into_iter().enumerate() {
            if i <= chunk.len() {
                out.push(char::from(ALPHABET[((n >> shift) & 63) as usize]));
            } else {
                out.push('=');
            }
        }
    }
    out
}

fn png_data_uri(bytes: &[u8]) -> String {
    format!("data:image/png;base64,{}", base64(bytes))
}

/// A 256 x 192 RGB PNG (red = x, green = a vertical ramp, a blue
/// checkerboard of 32-pixel squares): the golden fixture's geometry, so a
/// 6 x 8 merged grid of 48 rows. The synthetic cases use embedded images
/// only (the repository ignores `*.png` files).
const PATTERN_PNG_DATA_URI: &str =
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

/// A flat gray 256 x 192 PNG: the same geometry, different pixels.
const GRAY_PNG_DATA_URI: &str =
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAQAAAADACAAAAADOhuK6\
AAAA8UlEQVR42u3QMQEAAAwCIPsnM5Yhdg4ikD4XAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQI\
AAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAA\
AQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQ\
IECBAgQIAAAQIECBAgQIAAAQIECBAg4G7+IQjLAL6l/AAAAABJRU5ErkJggg==";

/// A 1 x 4000 grayscale strip: no small budget holds it without distorting
/// its aspect ratio.
const STRIP_PNG_DATA_URI: &str =
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAA+gCAAAAAAedin3\
AAAAH0lEQVR42u3CAQkAAAACoKY3vSGJpgEAAAAAAACAewNGt9Bqp3bDOgAAAABJRU5ErkJggg==";

// ─────────────────────────────────────────────────────────────────────────────
// The synthetic server
// ─────────────────────────────────────────────────────────────────────────────

/// `<|im_start|>` / `<|im_end|>` and the vision markers, right after the
/// 256 byte tokens.
const IM_START: u32 = 256;
const IM_END: u32 = 257;
const IDS: VisionTokenIds = VisionTokenIds {
    vision_start: 258,
    vision_end: 259,
    image_pad: 260,
};
/// The synthetic engine's KV window.
const MAX_SEQ: usize = 256;
/// The per-image budget of the synthetic service: the fixture's 48-row grid
/// fits, the strip does not.
const BUDGET: usize = 64;
const SEED: u64 = 7;
const MAX_TOKENS: usize = 4;
const PROMPT_TEXT: &str = "Describe this image briefly.";

/// The Bonsai 2 template's content rendering (string, or parts with image
/// parts as the vision placeholder) inside a minimal ChatML frame.
const TEMPLATE: &str = "{%- for message in messages %}\
{{- '<|im_start|>' + message.role + '\\n' }}\
{%- if message.content is string %}\
{{- message.content }}\
{%- else %}\
{%- for item in message.content %}\
{%- if 'image' in item or 'image_url' in item or item.type == 'image' %}\
{{- '<|vision_start|><|image_pad|><|vision_end|>' }}\
{%- elif 'text' in item %}\
{{- item.text }}\
{%- endif %}\
{%- endfor %}\
{%- endif %}\
{{- '<|im_end|>\\n' }}\
{%- endfor %}\
{%- if add_generation_prompt %}\
{{- '<|im_start|>assistant\\n' }}\
{%- endif %}";

/// GPT-2's byte -> printable-character map (the ByteLevel alphabet).
fn byte_level_alphabet() -> Vec<char> {
    let mut printable: Vec<u32> = ('!' as u32..='~' as u32).collect();
    printable.extend('¡' as u32..='¬' as u32);
    printable.extend('®' as u32..='ÿ' as u32);
    let mut out = Vec::with_capacity(256);
    let mut next_extra = 256u32;
    for byte in 0u32..256 {
        if printable.contains(&byte) {
            out.push(char::from_u32(byte).unwrap_or('?'));
        } else {
            out.push(char::from_u32(next_extra).unwrap_or('?'));
            next_extra += 1;
        }
    }
    out
}

/// A byte-level BPE `tokenizer.json` covering the synthetic model's whole
/// vocabulary: 256 byte tokens, the five markers as special added tokens,
/// and plain ASCII fillers up to `VOCAB` so every id the model can emit
/// decodes.
fn tokenizer_json() -> String {
    let mut vocab = serde_json::Map::new();
    for (id, ch) in byte_level_alphabet().into_iter().enumerate() {
        vocab.insert(ch.to_string(), Value::from(id as u64));
    }
    let specials = [
        (IM_START, "<|im_start|>"),
        (IM_END, "<|im_end|>"),
        (IDS.vision_start, "<|vision_start|>"),
        (IDS.vision_end, "<|vision_end|>"),
        (IDS.image_pad, "<|image_pad|>"),
    ];
    let first_filler = IDS.image_pad as usize + 1;
    for id in first_filler..VOCAB {
        vocab.insert(format!("<x{id}>"), Value::from(id as u64));
    }
    let added: Vec<Value> = specials
        .iter()
        .map(|(id, content)| json!({ "id": id, "content": content, "special": true }))
        .collect();
    json!({
        "model": { "type": "BPE", "vocab": vocab, "merges": [] },
        "added_tokens": added,
        "pre_tokenizer": { "type": "ByteLevel" },
        "decoder": { "type": "ByteLevel" }
    })
    .to_string()
}

fn template() -> ResolvedChatTemplate {
    ResolvedChatTemplate::Jinja(Arc::new(
        JinjaTemplate::compile(TEMPLATE).expect("the compact template compiles"),
    ))
}

fn tokenizer() -> TokenizerBridge {
    TokenizerBridge::native_from_json_str(&tokenizer_json())
        .expect("the synthetic tokenizer loads")
        .with_chat_template(template())
}

/// The synthetic `qwen35` GGUF, parsed once and kept for the process.
fn synthetic_gguf() -> &'static GgufFile<'static> {
    static GGUF: OnceLock<&'static GgufFile<'static>> = OnceLock::new();
    GGUF.get_or_init(|| {
        let bytes: &'static [u8] = Box::leak(synthetic_qwen35_gguf().into_boxed_slice());
        Box::leak(Box::new(
            GgufFile::parse(bytes).expect("the synthetic qwen35 fixture parses"),
        ))
    })
}

fn greedy() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        ..SamplingParams::default()
    }
}

fn synthetic_engine(max_seq: usize) -> InferenceEngine<'static> {
    InferenceEngine::from_gguf_with_backend(synthetic_gguf(), greedy(), SEED, max_seq, Backend::Cpu)
        .expect("the synthetic hybrid engine loads")
}

/// The tiny projector, widened to the synthetic model's hidden width.
fn synthetic_service(budget: usize) -> Arc<VisionService> {
    let spec = MmprojFixtureSpec {
        projection_dim: HIDDEN,
        ..MmprojFixtureSpec::tiny()
    };
    let bytes = synthetic_mmproj_gguf(&spec).expect("the synthetic projector builds");
    let gguf = GgufFile::parse(&bytes).expect("the synthetic projector parses");
    let tower = VisionTower::from_mmproj(&gguf).expect("the tiny tower loads");
    Arc::new(
        VisionService::new(tower, budget, ImageSourcePolicy::server(None, false))
            .expect("a valid budget")
            .with_token_ids(IDS),
    )
}

fn router_with(max_seq: usize, vision: Option<Arc<VisionService>>) -> axum::Router {
    let router = create_router(synthetic_engine(max_seq), Some(tokenizer()));
    match vision {
        Some(service) => router.layer(axum::Extension(service)),
        None => router,
    }
}

/// A router built the way `oxibonsai serve` builds one: the engine's KV
/// window is also the prompt ceiling (`max_input_tokens`).
fn serve_like_router(max_seq: usize, vision: Option<Arc<VisionService>>) -> axum::Router {
    let router = create_router_full(
        EnginePool::new(vec![synthetic_engine(max_seq)]),
        Some(tokenizer()),
        Arc::new(InferenceMetrics::new()),
        RouterOptions::default()
            .with_limits(RequestLimits::default().with_max_input_tokens(Some(max_seq))),
    );
    match vision {
        Some(service) => router.layer(axum::Extension(service)),
        None => router,
    }
}

fn vision_router() -> axum::Router {
    router_with(MAX_SEQ, Some(synthetic_service(BUDGET)))
}

fn text_router() -> axum::Router {
    router_with(MAX_SEQ, None)
}

// ─────────────────────────────────────────────────────────────────────────────
// Requests
// ─────────────────────────────────────────────────────────────────────────────

fn image_part(url: &str) -> Value {
    json!({ "type": "image_url", "image_url": { "url": url } })
}

fn text_part(text: &str) -> Value {
    json!({ "type": "text", "text": text })
}

/// A greedy chat request with one user message of `content`.
fn chat_body(content: Value) -> Value {
    json!({
        "messages": [{ "role": "user", "content": content }],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
    })
}

fn image_body(url: &str) -> Value {
    chat_body(json!([image_part(url), text_part(PROMPT_TEXT)]))
}

async fn send(app: &axum::Router, path: &str, body: &Value) -> (StatusCode, Vec<u8>) {
    let request = Request::post(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("build the request");
    let response = app
        .clone()
        .oneshot(request)
        .await
        .expect("the router answers");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read the body");
    (status, bytes.to_vec())
}

async fn post(app: &axum::Router, path: &str, body: &Value) -> (StatusCode, Value) {
    let (status, bytes) = send(app, path, body).await;
    let json = serde_json::from_slice(&bytes).unwrap_or_else(|e| {
        panic!(
            "{path}: not JSON ({e}): {}",
            String::from_utf8_lossy(&bytes)
        )
    });
    (status, json)
}

async fn chat(app: &axum::Router, body: &Value) -> Value {
    let (status, json) = post(app, "/v1/chat/completions", body).await;
    assert_eq!(status, StatusCode::OK, "{json}");
    json
}

/// The part of a chat answer that must be reproducible: the choices and
/// the usage (the id, timestamps and fingerprint are per response).
fn answer(json: &Value) -> Value {
    json!({ "choices": json["choices"], "usage": json["usage"] })
}

fn error_code(json: &Value) -> &str {
    json["error"]["code"].as_str().unwrap_or("<no error.code>")
}

async fn expect_refusal(app: &axum::Router, body: &Value, code: &str) {
    let (status, json) = post(app, "/v1/chat/completions", body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{code}: {json}");
    assert_eq!(error_code(&json), code, "{json}");
}

/// The rendered, tokenized prompt of `image_body` (one image, then the
/// text), exactly as the server renders it.
fn rendered_image_prompt_tokens(tok: &TokenizerBridge) -> Vec<u32> {
    let message = RenderMessage::with_parts(
        "user",
        vec![
            RenderContentPart::Image,
            RenderContentPart::Text(PROMPT_TEXT.to_string()),
        ],
    );
    let opts = RenderOptions {
        add_generation_prompt: true,
        ..RenderOptions::default()
    };
    let rendered = template()
        .render_with(&[message], &opts)
        .expect("the image prompt renders");
    tok.encode(&rendered).expect("the image prompt encodes")
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-6 * a.abs().max(b.abs()).max(1.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. The image round trip
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn a_data_uri_image_round_trips_as_the_engines_own_multimodal_generation() {
    let service = synthetic_service(BUDGET);
    let app = router_with(MAX_SEQ, Some(Arc::clone(&service)));
    let uri = PATTERN_PNG_DATA_URI.to_string();
    let mut body = image_body(&uri);
    body["logprobs"] = json!(true);
    body["top_logprobs"] = json!(2);
    let served = chat(&app, &body).await;

    // The same prompt, straight through the engine.
    let tok = tokenizer();
    let tokens = rendered_image_prompt_tokens(&tok);
    assert_eq!(
        tokens.iter().filter(|&&t| t == IDS.image_pad).count(),
        1,
        "one placeholder for the one image"
    );
    let images = service
        .encode_all(std::slice::from_ref(&uri))
        .expect("the fixture encodes");
    assert_eq!((images[0].grid.h, images[0].grid.w), (6, 8));
    let prompt = ChatPrompt::Multimodal(
        MultimodalPrompt::new(tokens.clone(), images, IDS).expect("the prompt splices"),
    );
    assert_eq!(prompt.len(), tokens.len() - 1 + 48);
    let mut engine = synthetic_engine(MAX_SEQ);
    let (ids, logprobs) = prompt
        .generate_with_logprobs(&mut engine, MAX_TOKENS, 2, &|id| format!("<{id}>"))
        .expect("the engine generates");
    assert!(!ids.is_empty(), "a comparison over at least one step");

    assert_eq!(served["usage"]["prompt_tokens"], json!(prompt.len()));
    assert_eq!(served["usage"]["completion_tokens"], json!(ids.len()));
    let content = served["choices"][0]["logprobs"]["content"]
        .as_array()
        .expect("per-token logprobs");
    assert_eq!(content.len(), logprobs.len());
    for (step, (got, want)) in content.iter().zip(&logprobs).enumerate() {
        let got = got["logprob"].as_f64().expect("a logprob");
        assert!(
            close(got, f64::from(want.logprob)),
            "step {step}: served {got}, engine {}",
            want.logprob
        );
    }

    // The extended endpoint serves the same prompt.
    let (status, extended) = post(&app, "/v1/chat/completions/extended", &image_body(&uri)).await;
    assert_eq!(status, StatusCode::OK, "{extended}");
    assert_eq!(extended["usage"]["prompt_tokens"], json!(prompt.len()));
    assert_eq!(extended["usage"]["completion_tokens"], json!(ids.len()));
}

#[tokio::test]
async fn a_different_image_changes_the_first_step_distribution() {
    let app = vision_router();
    let first_logprob = |json: &Value| {
        json["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            .as_array()
            .map(|alts| {
                alts.iter()
                    .filter_map(|a| a["logprob"].as_f64())
                    .collect::<Vec<f64>>()
            })
            .unwrap_or_default()
    };
    let mut fixture = image_body(PATTERN_PNG_DATA_URI);
    fixture["logprobs"] = json!(true);
    fixture["top_logprobs"] = json!(5);
    let mut gray = image_body(GRAY_PNG_DATA_URI);
    gray["logprobs"] = json!(true);
    gray["top_logprobs"] = json!(5);

    let a = chat(&app, &fixture).await;
    let b = chat(&app, &gray).await;
    assert_eq!(
        a["usage"]["prompt_tokens"], b["usage"]["prompt_tokens"],
        "same geometry, same row count"
    );
    let (a_lp, b_lp) = (first_logprob(&a), first_logprob(&b));
    assert_eq!(a_lp.len(), 5);
    assert_eq!(b_lp.len(), 5);
    assert_ne!(a_lp, b_lp, "the image rows reach the model");
    // ...and the same image again reproduces its answer on the same engine.
    assert_eq!(answer(&chat(&app, &fixture).await), answer(&a));
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Text-only requests are byte-unchanged
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn text_only_requests_are_byte_unchanged_by_the_projector() {
    let plain = chat_body(json!("Hello there, how are you?"));
    let parts = chat_body(json!([
        text_part("Hello there, "),
        text_part("how are you?")
    ]));

    let without = answer(&chat(&text_router(), &plain).await);
    let with = answer(&chat(&vision_router(), &plain).await);
    assert_eq!(with, without, "loading a projector changes no text answer");
    let as_parts = answer(&chat(&vision_router(), &parts).await);
    assert_eq!(as_parts, without, "text parts render as the string does");

    // The rendered prompt is the text itself: no vision marker anywhere.
    let tok = tokenizer();
    let rendered = template()
        .render_with(
            &[RenderMessage::new("user", "Hello there, how are you?")],
            &RenderOptions {
                add_generation_prompt: true,
                ..RenderOptions::default()
            },
        )
        .expect("renders");
    let tokens = tok.encode(&rendered).expect("encodes");
    assert!(!tokens.contains(&IDS.image_pad) && !tokens.contains(&IDS.vision_start));
    assert_eq!(without["usage"]["prompt_tokens"], json!(tokens.len()));
}

/// The pool reuses one engine across requests: an image request must leave
/// nothing behind that a later request could see.
#[tokio::test]
async fn requests_on_a_reused_engine_answer_like_a_fresh_engine() {
    let text = chat_body(json!("Tell me a story."));
    let image = image_body(PATTERN_PNG_DATA_URI);
    let gray = image_body(GRAY_PNG_DATA_URI);

    let fresh_text = answer(&chat(&vision_router(), &text).await);
    let fresh_image = answer(&chat(&vision_router(), &image).await);
    let fresh_gray = answer(&chat(&vision_router(), &gray).await);

    let app = vision_router();
    assert_eq!(answer(&chat(&app, &image).await), fresh_image);
    assert_eq!(
        answer(&chat(&app, &text).await),
        fresh_text,
        "text after an image"
    );
    assert_eq!(
        answer(&chat(&app, &image).await),
        fresh_image,
        "an image after text"
    );
    assert_eq!(
        answer(&chat(&app, &gray).await),
        fresh_gray,
        "an image after an image"
    );
    assert_eq!(
        answer(&chat(&app, &text).await),
        fresh_text,
        "text after two images"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Refusals, by name
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn every_refusal_carries_its_named_code() {
    let fixture = PATTERN_PNG_DATA_URI.to_string();

    // No projector loaded.
    expect_refusal(&text_router(), &image_body(&fixture), "vision_unavailable").await;
    let (status, json) = post(
        &text_router(),
        "/v1/chat/completions/extended",
        &image_body(&fixture),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{json}");
    assert_eq!(error_code(&json), "vision_unavailable");

    let app = vision_router();
    expect_refusal(
        &app,
        &image_body("data:image/png;base64,@@@@"),
        "image_data_uri_invalid",
    )
    .await;
    expect_refusal(
        &app,
        &image_body("https://example.com/cat.png"),
        "image_url_fetch_disabled",
    )
    .await;
    // A server resolves no local file without a media directory.
    expect_refusal(&app, &image_body("images/cat.png"), "image_file_refused").await;
    expect_refusal(
        &app,
        &image_body(STRIP_PNG_DATA_URI),
        "image_too_many_tokens",
    )
    .await;
    let many: Vec<Value> = (0..17)
        .map(|_| image_part(&fixture))
        .chain(std::iter::once(text_part(PROMPT_TEXT)))
        .collect();
    expect_refusal(&app, &chat_body(Value::Array(many)), "too_many_images").await;

    // The image rows count against every budget, on both endpoints, before
    // any image is encoded.
    let rows = rendered_image_prompt_tokens(&tokenizer()).len() - 1 + 48;
    assert!(rows > 80, "{rows}");
    let small = serve_like_router(80, Some(synthetic_service(BUDGET)));
    let text_alone = chat_body(json!([text_part(PROMPT_TEXT)]));
    let mut long = image_body(&fixture);
    let max_tokens = CONTEXT_LENGTH - rows + 1;
    long["max_tokens"] = json!(max_tokens);
    assert!(rows - 48 + max_tokens <= CONTEXT_LENGTH);
    for path in ["/v1/chat/completions", "/v1/chat/completions/extended"] {
        // With the KV window as the prompt ceiling (how `oxibonsai serve`
        // builds its router), the image prompt's rows cannot fit an
        // 80-position window while the same text alone does...
        let (status, json) = post(&small, path, &image_body(&fixture)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{path}: {json}");
        assert_eq!(
            error_code(&json),
            "max_input_tokens_exceeded",
            "{path}: {json}"
        );
        assert_eq!(
            json["error"]["n_prompt_tokens"],
            json!(rows),
            "{path}: {json}"
        );
        let (status, json) = post(&small, path, &text_alone).await;
        assert_eq!(
            status,
            StatusCode::OK,
            "{path}: the text alone fits: {json}"
        );

        // ...and against the model's declared context: rows + max_tokens
        // past it is refused although the text tokens alone would fit.
        let (status, json) = post(&app, path, &long).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{path}: {json}");
        assert_eq!(
            error_code(&json),
            "context_length_exceeded",
            "{path}: {json}"
        );
        assert_eq!(
            json["error"]["n_prompt_tokens"],
            json!(rows),
            "{path}: {json}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Streaming
// ─────────────────────────────────────────────────────────────────────────────

/// Every `data:` payload of an SSE body except the closing `[DONE]`.
fn sse_payloads(body: &[u8]) -> Vec<Value> {
    String::from_utf8_lossy(body)
        .split("\n\n")
        .filter_map(|event| {
            event
                .lines()
                .find_map(|line| line.strip_prefix("data: "))
                .filter(|data| *data != "[DONE]")
                .and_then(|data| serde_json::from_str(data).ok())
        })
        .collect()
}

/// The concatenated `field` deltas of a stream's first choice (`content` or
/// `reasoning_content`; a chunk without one contributes nothing).
fn streamed_text(payloads: &[Value], field: &str) -> String {
    payloads
        .iter()
        .filter_map(|p| p["choices"][0]["delta"][field].as_str())
        .collect()
}

/// A non-streamed answer's `field` text (`null` / absent is empty, as a
/// stream that sends no such delta is).
fn message_text(whole: &Value, field: &str) -> String {
    whole["choices"][0]["message"][field]
        .as_str()
        .unwrap_or_default()
        .to_string()
}

#[tokio::test]
async fn streaming_an_image_request_reports_the_non_streaming_usage() {
    let app = vision_router();
    let uri = PATTERN_PNG_DATA_URI.to_string();
    // The rendered prompt with its one placeholder replaced by the image's
    // 48 rows (a 6 x 8 merged grid).
    let rows = rendered_image_prompt_tokens(&tokenizer()).len() - 1 + 48;

    let mut body = image_body(&uri);
    body["stream"] = json!(true);
    body["stream_options"] = json!({ "include_usage": true });

    // Both chat endpoints stream an image request (SV-11), each delivering
    // exactly its own non-streaming answer (greedy, so seed-independent).
    for path in ["/v1/chat/completions", "/v1/chat/completions/extended"] {
        let (status, whole) = post(&app, path, &image_body(&uri)).await;
        assert_eq!(status, StatusCode::OK, "{path}: {whole}");
        assert_eq!(
            whole["usage"]["prompt_tokens"],
            json!(rows),
            "{path}: {whole}"
        );

        let (status, bytes) = send(&app, path, &body).await;
        assert_eq!(
            status,
            StatusCode::OK,
            "{path}: {}",
            String::from_utf8_lossy(&bytes)
        );
        let payloads = sse_payloads(&bytes);
        let usage = payloads
            .iter()
            .rev()
            .find(|p| p["usage"].is_object())
            .map(|p| p["usage"].clone())
            .unwrap_or_else(|| panic!("{path}: a usage chunk"));
        assert_eq!(
            usage["prompt_tokens"],
            json!(rows),
            "{path}: the usage counts the image's 48 rows"
        );
        assert_eq!(
            usage["completion_tokens"], whole["usage"]["completion_tokens"],
            "{path}"
        );
        assert!(
            usage["completion_tokens"].as_u64().is_some_and(|n| n > 0),
            "{path}: a comparison over at least one generated token: {usage}"
        );
        let finish = payloads
            .iter()
            .rev()
            .find_map(|p| p["choices"][0]["finish_reason"].as_str())
            .map(str::to_string);
        assert_eq!(
            finish.as_deref(),
            whole["choices"][0]["finish_reason"].as_str(),
            "{path}: the same stop"
        );
        assert_eq!(
            streamed_text(&payloads, "content"),
            message_text(&whole, "content"),
            "{path}: the streamed text is the non-streamed answer"
        );
        assert_eq!(
            streamed_text(&payloads, "reasoning_content"),
            message_text(&whole, "reasoning_content"),
            "{path}: the same reasoning split"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The real Bonsai 2 27B
// ─────────────────────────────────────────────────────────────────────────────

const PQ2_ENV: &str = "OXI_BONSAI2_PQ2_GGUF";
const MMPROJ_ENV: &str = "OXI_BONSAI2_MMPROJ_GGUF";
const MODELS_DIR_ENV: &str = "OXIBONSAI_MODELS_DIR";
const REQUIRE_ENV: &str = "OXI_REQUIRE_MODEL_FILES";
const PQ2_FILE: &str = "Ternary-Bonsai-2-27B-PQ2_0.gguf";
const MMPROJ_FILE: &str = "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf";
const CAPABILITY: &str = "bonsai2-vision";
const REAL_TEST: &str = "oxibonsai-runtime::bonsai2_vision_runtime_tests::\
                         real_27b_server_round_trip_reproduces_the_golden_prompt_one";
/// The real engine's KV window: the 67-row prompt plus 32 tokens, with room.
const REAL_KV_WINDOW: usize = 512;

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var(name)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .map(PathBuf::from)
}

fn locate(env: &str, file: &str) -> Option<PathBuf> {
    env_path(env)
        .or_else(|| env_path(MODELS_DIR_ENV).map(|dir| dir.join(file)))
        .filter(|p| p.is_file())
        .or_else(|| oxibonsai_testkit::workspace::find_model(file))
}

/// A line on the process's standard error that libtest does not capture.
fn report(line: &str) {
    let mut stderr = std::io::stderr().lock();
    // Bookkeeping only: a failed diagnostic write must not fail the test.
    let _ = writeln!(stderr, "{line}");
}

/// Append one `{"capability", "executed", "test", "duration_ms"}` record to
/// the capability manifest in one `write_all` (the test kit's schema).
fn record(executed: bool, duration: Option<std::time::Duration>) {
    let mut line = json!({
        "capability": CAPABILITY,
        "executed": executed,
        "test": REAL_TEST,
    });
    if let Some(d) = duration {
        line["duration_ms"] = json!(d.as_millis() as u64);
    }
    let path = oxibonsai_testkit::capability::report_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut text = line.to_string();
    text.push('\n');
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        let _ = file.write_all(text.as_bytes());
    }
    report(&format!(
        "CAPABILITY-REPORT capability={CAPABILITY} executed={executed} test={REAL_TEST}{}",
        duration.map_or_else(String::new, |d| format!(" duration_ms={}", d.as_millis()))
    ));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn real_27b_server_round_trip_reproduces_the_golden_prompt_one() {
    let (Some(model_path), Some(mmproj_path)) =
        (locate(PQ2_ENV, PQ2_FILE), locate(MMPROJ_ENV, MMPROJ_FILE))
    else {
        assert!(
            !std::env::var(REQUIRE_ENV).is_ok_and(|v| v.trim() == "1"),
            "{REQUIRE_ENV}=1: {REAL_TEST} needs {PQ2_FILE} and {MMPROJ_FILE} (set {PQ2_ENV} / \
             {MMPROJ_ENV})"
        );
        report(&format!(
            "skip {REAL_TEST}: set {PQ2_ENV} and {MMPROJ_ENV} (or {MODELS_DIR_ENV})"
        ));
        record(false, None);
        return;
    };
    let started = Instant::now();

    let golden: Value = serde_json::from_slice(
        &std::fs::read(golden_dir().join("vision.prompt1.cpu.json")).expect("the golden"),
    )
    .expect("golden JSON");
    let golden_text = golden["response"]["choices"][0]["message"]["content"]
        .as_str()
        .expect("golden content")
        .to_string();
    let golden_prompt_tokens = golden["response"]["usage"]["prompt_tokens"]
        .as_u64()
        .expect("golden prompt_tokens");
    assert_eq!(golden_prompt_tokens, 67, "the golden's own usage");

    // The language model: mapped, never read into memory.
    let mmap: &'static memmap2::Mmap = Box::leak(Box::new(
        mmap_gguf_file(&model_path).expect("map the 27B GGUF"),
    ));
    let gguf: &'static GgufFile<'static> = Box::leak(Box::new(
        GgufFile::parse(mmap).expect("the 27B GGUF parses"),
    ));
    let loading = Instant::now();
    let engine =
        InferenceEngine::from_gguf_with_backend(gguf, greedy(), 42, REAL_KV_WINDOW, Backend::Cpu)
            .expect("the 27B loads as a CPU hybrid engine");
    let template = ResolvedChatTemplate::from_gguf(&gguf.metadata).expect("its chat template");
    let tok = oxibonsai_runtime::engine::tokenizer_from_gguf(gguf)
        .expect("its embedded tokenizer")
        .with_chat_template(template);
    report(&format!(
        "bonsai2-vision server: language model loaded in {:.2}s",
        loading.elapsed().as_secs_f64()
    ));

    let loading = Instant::now();
    let service = VisionService::load(
        &mmproj_path,
        oxibonsai_model::vision::DEFAULT_IMAGE_MAX_TOKENS,
        ImageSourcePolicy::server(None, false),
    )
    .expect("the projector loads");
    report(&format!(
        "bonsai2-vision server: projector loaded in {:.2}s",
        loading.elapsed().as_secs_f64()
    ));
    let app = create_router(engine, Some(tok)).layer(axum::Extension(Arc::new(service)));

    let body = json!({
        "messages": [{
            "role": "user",
            "content": [
                image_part(&png_data_uri(&golden_fixture_png())),
                text_part(PROMPT_TEXT),
            ],
        }],
        "max_tokens": 32,
        "temperature": 0.0,
        "chat_template_kwargs": { "enable_thinking": false },
    });
    let request_started = Instant::now();
    let (status, json) = post(&app, "/v1/chat/completions", &body).await;
    report(&format!(
        "bonsai2-vision server: request (encode + prefill + 32-token decode) in {:.2}s",
        request_started.elapsed().as_secs_f64()
    ));
    assert_eq!(status, StatusCode::OK, "{json}");
    let content = json["choices"][0]["message"]["content"]
        .as_str()
        .expect("an answer")
        .to_string();
    report(&format!("bonsai2-vision server answer: {content:?}"));
    assert_eq!(json["usage"]["prompt_tokens"], json!(golden_prompt_tokens));
    assert_eq!(json["usage"]["completion_tokens"], json!(32));
    assert_eq!(json["choices"][0]["finish_reason"], json!("length"));
    assert_eq!(
        content, golden_text,
        "the fork's golden answer, byte for byte"
    );
    report(&format!(
        "bonsai2-vision server golden prompt1: text identical ({} bytes), 67 prompt tokens \
         (48 image rows), 32 completion tokens",
        golden_text.len()
    ));
    record(true, Some(started.elapsed()));
}
