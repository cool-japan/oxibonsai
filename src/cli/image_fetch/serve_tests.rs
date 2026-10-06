//! The fetcher behind the real chat endpoints, end to end and hermetic: a
//! router over the test kit's synthetic `qwen35` engine with the tiny
//! projector widened to its hidden width, a byte-level tokenizer carrying the
//! chat and vision markers, a compact template that renders an image part as
//! the Bonsai 2 template does, and `image_url`s that point at a loopback
//! listener — allowlisted, not allowlisted, stalling.

use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use tower::ServiceExt;

use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::gguf::writer::GgufWriter;
use oxibonsai_core::MetadataWriteValue;
use oxibonsai_model::vision::{ImageSourcePolicy, VisionTokenIds, VisionTower};
use oxibonsai_runtime::config::ResolvedChatTemplate;
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::engine_pool::EnginePool;
use oxibonsai_runtime::engine_seam::Backend;
use oxibonsai_runtime::metrics::InferenceMetrics;
use oxibonsai_runtime::sampling::SamplingParams;
use oxibonsai_runtime::server::{create_router_full, RequestLimits, RouterOptions};
use oxibonsai_runtime::vision_prefill::VisionService;
use oxibonsai_runtime::TokenizerBridge;
use oxibonsai_testkit::mmproj_fixture::{synthetic_mmproj_gguf, MmprojFixtureSpec};
use oxibonsai_testkit::qwen35_fixture::{synthetic_qwen35_gguf, HIDDEN, VOCAB};

use super::capacity::{FetchCapacity, FetchCounts, MAX_FETCHES_IN_FLIGHT, MAX_FETCHES_WAITING};
use super::test_server::{Reply, TestServer};
use super::{ImageFetchSettings, ImageUrlFetcher};
use crate::cli::bonsai2::tests::PATTERN_PNG_DATA_URI;

// ── The synthetic server ─────────────────────────────────────────────────

const IM_START: u32 = 256;
const IM_END: u32 = 257;
const IDS: VisionTokenIds = VisionTokenIds {
    vision_start: 258,
    vision_end: 259,
    image_pad: 260,
};
const MAX_SEQ: usize = 256;
const BUDGET: usize = 64;
const SEED: u64 = 7;
const PROMPT_TEXT: &str = "Describe this image briefly.";

/// The Bonsai 2 template's content rendering inside a minimal ChatML frame.
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

/// GPT-2's byte -> printable-character map.
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
    for id in (IDS.image_pad as usize + 1)..VOCAB {
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

/// The compact template, compiled the way a model's own
/// `tokenizer.chat_template` is.
fn template() -> ResolvedChatTemplate {
    let mut writer = GgufWriter::new();
    writer.add_metadata(
        "general.architecture",
        MetadataWriteValue::Str("qwen35".to_string()),
    );
    writer.add_metadata(
        "tokenizer.chat_template",
        MetadataWriteValue::Str(TEMPLATE.to_string()),
    );
    let bytes = writer.to_bytes().expect("serialize the template carrier");
    let gguf = GgufFile::parse(&bytes).expect("parse the template carrier");
    ResolvedChatTemplate::from_gguf(&gguf.metadata).expect("the compact template compiles")
}

fn tokenizer() -> TokenizerBridge {
    TokenizerBridge::native_from_json_str(&tokenizer_json())
        .expect("the synthetic tokenizer loads")
        .with_chat_template(template())
}

fn synthetic_gguf() -> &'static GgufFile<'static> {
    static GGUF: OnceLock<&'static GgufFile<'static>> = OnceLock::new();
    GGUF.get_or_init(|| {
        let bytes: &'static [u8] = Box::leak(synthetic_qwen35_gguf().into_boxed_slice());
        Box::leak(Box::new(
            GgufFile::parse(bytes).expect("the synthetic qwen35 fixture parses"),
        ))
    })
}

fn engine() -> InferenceEngine<'static> {
    let greedy = SamplingParams {
        temperature: 0.0,
        ..SamplingParams::default()
    };
    InferenceEngine::from_gguf_with_backend(synthetic_gguf(), greedy, SEED, MAX_SEQ, Backend::Cpu)
        .expect("the synthetic hybrid engine loads")
}

/// The tiny projector at the synthetic model's width, resolving references
/// under `policy`.
fn service(policy: ImageSourcePolicy) -> Arc<VisionService> {
    let spec = MmprojFixtureSpec {
        projection_dim: HIDDEN,
        ..MmprojFixtureSpec::tiny()
    };
    let bytes = synthetic_mmproj_gguf(&spec).expect("the synthetic projector builds");
    let gguf = GgufFile::parse(&bytes).expect("the synthetic projector parses");
    let tower = VisionTower::from_mmproj(&gguf).expect("the tiny tower loads");
    Arc::new(
        VisionService::new(tower, BUDGET, policy)
            .expect("a valid budget")
            .with_token_ids(IDS),
    )
}

/// A server whose requests must finish within `timeout_ms` (`0`: none).
fn router(policy: ImageSourcePolicy, timeout_ms: u64) -> axum::Router {
    create_router_full(
        EnginePool::new(vec![engine()]),
        Some(tokenizer()),
        Arc::new(InferenceMetrics::new()),
        RouterOptions::default().with_limits(RequestLimits::default().with_timeout_ms(timeout_ms)),
    )
    .layer(axum::Extension(service(policy)))
}

/// The policy `serve --allow-image-url-fetch` builds, with these settings —
/// under transfer slots of the test's own, sized like the process-wide
/// bound (tests run side by side in one process).
fn fetching(timeout_ms: u64, allow: &[String]) -> ImageSourcePolicy {
    fetching_under(
        timeout_ms,
        allow,
        FetchCapacity::new(MAX_FETCHES_IN_FLIGHT, MAX_FETCHES_WAITING),
    )
}

/// [`fetching`] under `capacity`.
fn fetching_under(
    timeout_ms: u64,
    allow: &[String],
    capacity: Arc<FetchCapacity>,
) -> ImageSourcePolicy {
    let settings =
        ImageFetchSettings::resolve(Some(timeout_ms), None, allow, None).expect("valid settings");
    ImageSourcePolicy::server(None, true).with_remote_fetcher(
        ImageUrlFetcher::new(settings)
            .with_capacity(capacity)
            .into_shared(),
    )
}

fn png() -> Vec<u8> {
    oxibonsai_model::vision::load_image_bytes(
        PATTERN_PNG_DATA_URI,
        &ImageSourcePolicy::local_user(),
    )
    .expect("the embedded PNG decodes")
}

fn image_body(urls: &[&str]) -> Value {
    let mut content: Vec<Value> = urls
        .iter()
        .map(|url| json!({ "type": "image_url", "image_url": { "url": url } }))
        .collect();
    content.push(json!({ "type": "text", "text": PROMPT_TEXT }));
    json!({
        "messages": [{ "role": "user", "content": content }],
        "max_tokens": 4,
        "temperature": 0.0,
    })
}

async fn post(app: &axum::Router, path: &str, body: &Value) -> (StatusCode, String) {
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
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

fn parse(text: &str) -> Value {
    serde_json::from_str(text).unwrap_or_else(|e| panic!("not JSON ({e}): {text}"))
}

/// The `usage.prompt_tokens` an SSE stream reports in its usage chunk.
fn streamed_prompt_tokens(sse: &str) -> Option<u64> {
    sse.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter_map(|data| serde_json::from_str::<Value>(data).ok())
        .find_map(|chunk| chunk["usage"]["prompt_tokens"].as_u64())
}

// ── The image arrives as if sent inline ──────────────────────────────────

/// An `http://127.0.0.1:<port>/img.png` the operator allowlisted answers like
/// the same bytes sent as a data URI — the same answer, the same
/// `prompt_tokens` — on both chat endpoints, streamed or not.
#[tokio::test]
async fn an_allowlisted_image_url_answers_like_its_data_uri() {
    let body = png();
    let images = TestServer::start("127.0.0.1", move |_| Reply::Ok(body.clone()));
    let url = images.url("/img.png");
    let allow = vec![format!("127.0.0.1:{}", images.port())];
    let app = router(fetching(5_000, &allow), 0);

    let (status, by_uri) = post(
        &app,
        "/v1/chat/completions",
        &image_body(&[PATTERN_PNG_DATA_URI]),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{by_uri}");
    let (status, by_url) = post(&app, "/v1/chat/completions", &image_body(&[url.as_str()])).await;
    assert_eq!(status, StatusCode::OK, "{by_url}");
    let (by_uri, by_url) = (parse(&by_uri), parse(&by_url));
    assert_eq!(by_url["choices"], by_uri["choices"], "the same answer");
    assert_eq!(by_url["usage"], by_uri["usage"], "the same usage");
    let prompt_tokens = by_uri["usage"]["prompt_tokens"]
        .as_u64()
        .expect("prompt_tokens");
    assert!(
        prompt_tokens > 48,
        "the image's 48 rows are counted: {prompt_tokens}"
    );

    let (status, extended) = post(
        &app,
        "/v1/chat/completions/extended",
        &image_body(&[url.as_str()]),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{extended}");
    assert_eq!(
        parse(&extended)["usage"]["prompt_tokens"].as_u64(),
        Some(prompt_tokens)
    );

    for path in ["/v1/chat/completions", "/v1/chat/completions/extended"] {
        let mut streamed = image_body(&[url.as_str()]);
        streamed["stream"] = json!(true);
        streamed["stream_options"] = json!({ "include_usage": true });
        let (status, sse) = post(&app, path, &streamed).await;
        assert_eq!(status, StatusCode::OK, "{path}: {sse}");
        assert_eq!(
            streamed_prompt_tokens(&sse),
            Some(prompt_tokens),
            "{path}: {sse}"
        );
    }
    assert_eq!(images.accepts(), 4, "one fetch per URL request");
}

// ── Refusals ─────────────────────────────────────────────────────────────

/// Without the opt-in a remote `image_url` is a `400
/// image_url_fetch_disabled`; opted in but not allowlisted, a loopback one is
/// a `400 image_url_refused` — and neither opens a connection.
#[tokio::test]
async fn refused_image_urls_are_typed_400s_that_open_nothing() {
    let images = TestServer::start("127.0.0.1", |_| Reply::Ok(Vec::new()));
    let url = images.url("/img.png");
    for (policy, code) in [
        (
            ImageSourcePolicy::server(None, false),
            "image_url_fetch_disabled",
        ),
        (
            ImageSourcePolicy::server(None, true),
            "image_url_fetch_disabled",
        ),
        (fetching(5_000, &[]), "image_url_refused"),
    ] {
        let app = router(policy, 0);
        for path in ["/v1/chat/completions", "/v1/chat/completions/extended"] {
            let (status, text) = post(&app, path, &image_body(&[url.as_str()])).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{path}: {text}");
            let json = parse(&text);
            assert_eq!(json["error"]["code"], code, "{path}: {json}");
            assert_eq!(json["error"]["param"], "messages", "{path}: {json}");
        }
    }
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(images.accepts(), 0, "nothing was opened");
}

/// An image server that stalls past the per-image deadline is a typed `400
/// image_url_fetch_failed` naming the deadline.
#[tokio::test]
async fn a_stalled_image_server_is_a_typed_fetch_failure() {
    let images = TestServer::start("127.0.0.1", |_| Reply::Stall(Duration::from_secs(10)));
    let allow = vec![format!("127.0.0.1:{}", images.port())];
    let app = router(fetching(300, &allow), 0);
    let started = Instant::now();
    let (status, text) = post(
        &app,
        "/v1/chat/completions",
        &image_body(&[images.url("/img.png").as_str()]),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{text}");
    let json = parse(&text);
    assert_eq!(json["error"]["code"], "image_url_fetch_failed", "{json}");
    assert!(
        json["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("timed out after 300 ms"),
        "{json}"
    );
    assert!(started.elapsed() < Duration::from_secs(5));
}

/// A server deadline that expires while an image is being fetched is a `504`
/// naming the `image_fetch` stage, on both chat endpoints; the fetch in
/// flight is dropped at once — the image server sees its connection closed
/// long before the fetch's own 10 s deadline — and the request's second
/// image is never fetched.
#[tokio::test]
async fn a_server_deadline_during_a_fetch_names_image_fetch_and_stops_fetching() {
    for path in ["/v1/chat/completions", "/v1/chat/completions/extended"] {
        let images = TestServer::start("127.0.0.1", |_| Reply::Stall(Duration::from_secs(30)));
        let allow = vec![format!("127.0.0.1:{}", images.port())];
        let capacity = FetchCapacity::new(MAX_FETCHES_IN_FLIGHT, MAX_FETCHES_WAITING);
        let app = router(fetching_under(10_000, &allow, Arc::clone(&capacity)), 400);
        let first = images.url("/first.png");
        let second = images.url("/second.png");
        let (status, text) =
            post(&app, path, &image_body(&[first.as_str(), second.as_str()])).await;
        let answered_at = Instant::now();
        assert_eq!(status, StatusCode::GATEWAY_TIMEOUT, "{path}: {text}");
        let json = parse(&text);
        assert_eq!(json["error"]["code"], "request_timeout", "{path}: {json}");
        assert_eq!(json["error"]["phase"], "image_fetch", "{path}: {json}");
        let message = json["error"]["message"].as_str().unwrap_or_default();
        assert!(
            message.contains("while fetching one of the request's remote images"),
            "{path}: {message}"
        );

        // The abandoned fetch closes its connection at once (the image
        // server sees it), its slot comes back, and nothing follows it.
        let (closed, accepts, counts) = tokio::task::spawn_blocking(move || {
            let closed = images.wait_finished(1, Duration::from_millis(2_000));
            std::thread::sleep(Duration::from_millis(200));
            (closed, images.accepts(), capacity.counts())
        })
        .await
        .expect("the waiting task");
        assert!(
            closed,
            "{path}: the fetch of an abandoned request must not run on to its own deadline"
        );
        assert!(
            answered_at.elapsed() < Duration::from_secs(5),
            "{path}: closed {:?} after the 504",
            answered_at.elapsed()
        );
        assert_eq!(accepts, 1, "{path}: the second image was never fetched");
        assert_eq!(
            counts,
            FetchCounts::default(),
            "{path}: every slot came back"
        );
    }
}

/// A client that goes away while its image is being fetched: the fetch is
/// dropped at once (the image server sees the connection closed, far below
/// the fetch's own 10 s deadline), on both chat endpoints.
#[tokio::test]
async fn a_client_that_goes_away_mid_fetch_closes_the_fetch_at_once() {
    for path in ["/v1/chat/completions", "/v1/chat/completions/extended"] {
        let images = Arc::new(TestServer::start("127.0.0.1", |_| {
            Reply::Stall(Duration::from_secs(30))
        }));
        let allow = vec![format!("127.0.0.1:{}", images.port())];
        let capacity = FetchCapacity::new(MAX_FETCHES_IN_FLIGHT, MAX_FETCHES_WAITING);
        let app = router(fetching_under(10_000, &allow, Arc::clone(&capacity)), 0);
        let body = image_body(&[images.url("/img.png").as_str()]);
        let request = Request::post(path)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .expect("build the request");
        let client = tokio::spawn(app.clone().oneshot(request));
        let connected = {
            let images = Arc::clone(&images);
            tokio::task::spawn_blocking(move || {
                let start = Instant::now();
                while images.accepts() == 0 && start.elapsed() < Duration::from_secs(10) {
                    std::thread::sleep(Duration::from_millis(2));
                }
                images.accepts() == 1
            })
            .await
            .expect("the waiting task")
        };
        assert!(connected, "{path}: the fetch never connected");
        // The client goes away: its handler future is dropped.
        client.abort();
        assert!(
            client.await.is_err(),
            "{path}: the request was still fetching"
        );
        let (closed, counts) = tokio::task::spawn_blocking(move || {
            let closed = images.wait_finished(1, Duration::from_millis(2_000));
            let start = Instant::now();
            while capacity.counts() != FetchCounts::default()
                && start.elapsed() < Duration::from_secs(5)
            {
                std::thread::sleep(Duration::from_millis(2));
            }
            (closed, capacity.counts())
        })
        .await
        .expect("the waiting task");
        assert!(
            closed,
            "{path}: our side must close the fetch's connection when the client goes away"
        );
        assert_eq!(
            counts,
            FetchCounts::default(),
            "{path}: every slot came back"
        );
    }
}

// ── The bound, over HTTP ─────────────────────────────────────────────────

/// A fetch the fetcher refuses because it is at capacity is a load shed, not
/// a fault in the request: `503`, `Retry-After`, `type: overloaded_error`,
/// `code: image_fetch_overloaded`, on both chat endpoints, streamed or not
/// (the answer comes before any stream opens) — and nothing is opened.
#[tokio::test]
async fn a_fetcher_at_capacity_answers_a_retryable_503() {
    let images = TestServer::start("127.0.0.1", |_| Reply::Ok(png()));
    let allow = vec![format!("127.0.0.1:{}", images.port())];
    // No slot at all: every fetch is over the bound.
    let app = router(fetching_under(5_000, &allow, FetchCapacity::new(0, 0)), 0);
    let url = images.url("/img.png");
    for path in ["/v1/chat/completions", "/v1/chat/completions/extended"] {
        for stream in [false, true] {
            let mut body = image_body(&[url.as_str()]);
            body["stream"] = json!(stream);
            let request = Request::post(path)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .expect("build the request");
            let response = app
                .clone()
                .oneshot(request)
                .await
                .expect("the router answers");
            assert_eq!(
                response.status(),
                StatusCode::SERVICE_UNAVAILABLE,
                "{path} {stream}"
            );
            assert_eq!(
                response
                    .headers()
                    .get("retry-after")
                    .and_then(|v| v.to_str().ok()),
                Some("1"),
                "{path} {stream}"
            );
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .expect("read the body");
            let json = parse(&String::from_utf8_lossy(&bytes));
            assert_eq!(json["error"]["type"], "overloaded_error", "{json}");
            assert_eq!(json["error"]["code"], "image_fetch_overloaded", "{json}");
            assert!(json["error"]["param"].is_null(), "{json}");
            let message = json["error"]["message"].as_str().unwrap_or_default();
            assert!(message.contains("at capacity"), "{message}");
            assert!(message.starts_with("image 0:"), "{message}");
        }
    }
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(images.accepts(), 0, "a refused fetch opens nothing");
}
