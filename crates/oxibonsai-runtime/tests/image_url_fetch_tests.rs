//! Remote `image_url`s through the OpenAI-compatible chat endpoints, at the
//! runtime's seam: the server resolves a remote reference only through the
//! fetcher its [`ImageSourcePolicy`] carries (`RemoteImageFetcher`), so these
//! tests install stub fetchers — one that serves bytes, one that refuses like
//! the address policy, one that fails like a stalled server, one that is
//! slow — and check what the server does with each:
//!
//! 1. a fetched image answers exactly like the same bytes sent as a data URI
//!    (the same answer, the same `usage.prompt_tokens`), on both chat
//!    endpoints, streamed or not; a data-URI request never calls the fetcher;
//! 2. a policy without the opt-in, or with the opt-in but no fetcher, is a
//!    `400 image_url_fetch_disabled` whose reason tells the two apart; a
//!    refusal is a `400 image_url_refused`, a failed fetch a `400
//!    image_url_fetch_failed`, each with `error.param = messages`;
//! 3. a per-request deadline that expires while an image is being fetched is
//!    a `504` whose `error.phase` is `image_fetch`, and the request's next
//!    image is never fetched;
//! 4. a client that goes away mid-fetch leaves no further fetch behind, on
//!    either endpoint;
//! 5. one request's images are fetched one after another, each once, and a
//!    request with more images than the per-request cap is refused before
//!    any of them is fetched;
//! 6. a fetch in flight learns that its request was abandoned (its client
//!    went away, or its deadline expired) through `CurrentFetch`, on either
//!    endpoint, so a fetcher can drop it at once; and a fetcher that refuses
//!    a fetch because it is at capacity gets the request a retryable `503`
//!    (`Retry-After`, `type: overloaded_error`, `code:
//!    image_fetch_overloaded`), streamed or not.
//!
//! The server is the synthetic one of `bonsai2_vision_runtime_tests.rs`: the
//! test kit's `qwen35` hybrid, its tiny projector widened to the model's
//! width, a byte-level tokenizer carrying the chat and vision markers, and a
//! compact template that renders an image part as the Bonsai 2 template does.
//! (The real fetcher, against loopback listeners, is tested where it lives:
//! the `oxibonsai` command's `cli::image_fetch` tests.)

#![cfg(feature = "server")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use tower::ServiceExt;

use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_model::vision::image_decode::MAX_ENCODED_IMAGE_BYTES;
use oxibonsai_model::vision::remote::{AddressClass, RemoteFetchFailure, RemoteUrlRefusal};
use oxibonsai_model::vision::{
    load_image_bytes, ImageInputError, ImageSourcePolicy, RemoteImageFetcher,
    SharedRemoteImageFetcher, VisionTokenIds, VisionTower,
};
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::engine_pool::EnginePool;
use oxibonsai_runtime::engine_seam::Backend;
use oxibonsai_runtime::metrics::InferenceMetrics;
use oxibonsai_runtime::sampling::SamplingParams;
use oxibonsai_runtime::server::{create_router_full, RequestLimits, RouterOptions};
use oxibonsai_runtime::vision_prefill::{CurrentFetch, VisionService, MAX_IMAGES_PER_REQUEST};
use oxibonsai_runtime::TokenizerBridge;
use oxibonsai_testkit::mmproj_fixture::{synthetic_mmproj_gguf, MmprojFixtureSpec};
use oxibonsai_testkit::qwen35_fixture::{synthetic_qwen35_gguf, HIDDEN, VOCAB};
use oxibonsai_tokenizer::chat_templates::ResolvedChatTemplate;
use oxibonsai_tokenizer::jinja::JinjaTemplate;

// ─────────────────────────────────────────────────────────────────────────────
// The synthetic server
// ─────────────────────────────────────────────────────────────────────────────

/// A 256 x 192 RGB PNG: the vision golden's geometry, a 6 x 8 merged grid.
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

fn tokenizer() -> TokenizerBridge {
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
    let json = json!({
        "model": { "type": "BPE", "vocab": vocab, "merges": [] },
        "added_tokens": added,
        "pre_tokenizer": { "type": "ByteLevel" },
        "decoder": { "type": "ByteLevel" }
    })
    .to_string();
    let template = ResolvedChatTemplate::Jinja(Arc::new(
        JinjaTemplate::compile(TEMPLATE).expect("the compact template compiles"),
    ));
    TokenizerBridge::native_from_json_str(&json)
        .expect("the synthetic tokenizer loads")
        .with_chat_template(template)
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

fn png() -> Vec<u8> {
    load_image_bytes(PATTERN_PNG_DATA_URI, &ImageSourcePolicy::local_user())
        .expect("the embedded PNG decodes")
}

// ─────────────────────────────────────────────────────────────────────────────
// Stub fetchers
// ─────────────────────────────────────────────────────────────────────────────

/// Answers every URL with `body` after `hold`, counting its fetches and the
/// most it was ever asked to run at once.
struct StubFetcher {
    body: Vec<u8>,
    hold: Duration,
    calls: AtomicUsize,
    in_flight: AtomicUsize,
    most_in_flight: AtomicUsize,
}

impl StubFetcher {
    fn serving(body: Vec<u8>, hold: Duration) -> Arc<Self> {
        Arc::new(Self {
            body,
            hold,
            calls: AtomicUsize::new(0),
            in_flight: AtomicUsize::new(0),
            most_in_flight: AtomicUsize::new(0),
        })
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    fn most_in_flight(&self) -> usize {
        self.most_in_flight.load(Ordering::SeqCst)
    }
}

impl RemoteImageFetcher for StubFetcher {
    fn fetch(&self, url: &str, max_bytes: usize) -> Result<Vec<u8>, ImageInputError> {
        assert!(url.starts_with("http"), "only remote references: {url}");
        assert_eq!(max_bytes, MAX_ENCODED_IMAGE_BYTES, "the encoded-image cap");
        let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.most_in_flight.fetch_max(now, Ordering::SeqCst);
        self.calls.fetch_add(1, Ordering::SeqCst);
        std::thread::sleep(self.hold);
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        Ok(self.body.clone())
    }
}

/// Refuses every URL the way the address policy refuses a loopback host.
struct Refusing;

impl RemoteImageFetcher for Refusing {
    fn fetch(&self, url: &str, _max_bytes: usize) -> Result<Vec<u8>, ImageInputError> {
        Err(RemoteUrlRefusal::NotPublic {
            host: "127.0.0.1".to_string(),
            class: Some(AddressClass::Loopback),
        }
        .into_error(url))
    }
}

/// Fails every URL the way a stalled server times out.
struct Failing;

impl RemoteImageFetcher for Failing {
    fn fetch(&self, url: &str, _max_bytes: usize) -> Result<Vec<u8>, ImageInputError> {
        Err(RemoteFetchFailure::TimedOut { ms: 300 }.into_error(url))
    }
}

fn fetching(fetcher: Arc<dyn RemoteImageFetcher>) -> ImageSourcePolicy {
    ImageSourcePolicy::server(None, true)
        .with_remote_fetcher(SharedRemoteImageFetcher::new(fetcher))
}

// ─────────────────────────────────────────────────────────────────────────────
// Requests
// ─────────────────────────────────────────────────────────────────────────────

const URL: &str = "https://images.example/cat.png";

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

fn request(path: &str, body: &Value) -> Request<Body> {
    Request::post(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("build the request")
}

async fn post(app: &axum::Router, path: &str, body: &Value) -> (StatusCode, String) {
    let response = app
        .clone()
        .oneshot(request(path, body))
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

const ENDPOINTS: [&str; 2] = ["/v1/chat/completions", "/v1/chat/completions/extended"];

// ─────────────────────────────────────────────────────────────────────────────
// 1. A fetched image is the same image
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn a_fetched_image_answers_like_its_data_uri() {
    let stub = StubFetcher::serving(png(), Duration::ZERO);
    let app = router(fetching(stub.clone()), 0);

    let (status, by_uri) = post(&app, ENDPOINTS[0], &image_body(&[PATTERN_PNG_DATA_URI])).await;
    assert_eq!(status, StatusCode::OK, "{by_uri}");
    assert_eq!(stub.calls(), 0, "a data URI never reaches the fetcher");
    let by_uri = parse(&by_uri);
    let prompt_tokens = by_uri["usage"]["prompt_tokens"]
        .as_u64()
        .expect("prompt_tokens");
    assert!(prompt_tokens > 48, "{prompt_tokens}");

    for path in ENDPOINTS {
        let (status, by_url) = post(&app, path, &image_body(&[URL])).await;
        assert_eq!(status, StatusCode::OK, "{path}: {by_url}");
        let by_url = parse(&by_url);
        assert_eq!(
            by_url["usage"]["prompt_tokens"].as_u64(),
            Some(prompt_tokens),
            "{path}"
        );
        if path == ENDPOINTS[0] {
            assert_eq!(by_url["choices"], by_uri["choices"], "the same answer");
            assert_eq!(by_url["usage"], by_uri["usage"]);
        }
        let mut streamed = image_body(&[URL]);
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
    assert_eq!(stub.calls(), 4, "one fetch per request");
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Typed refusals
// ─────────────────────────────────────────────────────────────────────────────

/// Without the opt-in, and with an opt-in nothing can act on, a remote
/// reference is a `400 image_url_fetch_disabled` — and the reasons differ.
#[tokio::test]
async fn a_remote_reference_without_a_fetcher_is_disabled_and_says_why() {
    let mut reasons = Vec::new();
    for policy in [
        ImageSourcePolicy::server(None, false),
        ImageSourcePolicy::server(None, true),
    ] {
        let app = router(policy, 0);
        for path in ENDPOINTS {
            let (status, text) = post(&app, path, &image_body(&[URL])).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{path}: {text}");
            let json = parse(&text);
            assert_eq!(json["error"]["code"], "image_url_fetch_disabled", "{json}");
            assert_eq!(json["error"]["param"], "messages", "{json}");
            reasons.push(
                json["error"]["message"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string(),
            );
        }
    }
    assert!(
        reasons[0].contains("--allow-image-url-fetch"),
        "{}",
        reasons[0]
    );
    assert!(
        reasons[2].contains("installed no fetcher"),
        "{}",
        reasons[2]
    );
    assert_ne!(reasons[0], reasons[2]);
}

/// The fetcher's refusal and its failure reach the client as their own
/// typed `400`s, with the image's index and `error.param = messages`.
#[tokio::test]
async fn a_refusal_and_a_failed_fetch_are_typed_400s() {
    let refusing: Arc<dyn RemoteImageFetcher> = Arc::new(Refusing);
    let failing: Arc<dyn RemoteImageFetcher> = Arc::new(Failing);
    for (fetcher, code, words) in [
        (refusing, "image_url_refused", "not a public address"),
        (failing, "image_url_fetch_failed", "timed out after 300 ms"),
    ] {
        let app = router(fetching(fetcher), 0);
        for path in ENDPOINTS {
            for stream in [false, true] {
                let mut body = image_body(&[URL]);
                body["stream"] = json!(stream);
                let (status, text) = post(&app, path, &body).await;
                assert_eq!(status, StatusCode::BAD_REQUEST, "{path} {stream}: {text}");
                let json = parse(&text);
                assert_eq!(json["error"]["code"], code, "{json}");
                assert_eq!(json["error"]["param"], "messages", "{json}");
                let message = json["error"]["message"].as_str().unwrap_or_default();
                assert!(message.contains(words), "{message}");
                assert!(message.starts_with("image 0:"), "{message}");
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. The deadline names the fetch, and the fetching stops
// ─────────────────────────────────────────────────────────────────────────────

/// A per-request deadline that expires while an image is being fetched is a
/// `504` naming the `image_fetch` stage, streamed or not; once the fetch in
/// flight ends, the request's second image is never fetched.
#[tokio::test]
async fn a_deadline_during_a_fetch_is_a_504_naming_image_fetch() {
    for stream in [false, true] {
        let stub = StubFetcher::serving(png(), Duration::from_millis(1_200));
        let app = router(fetching(stub.clone()), 300);
        let mut body = image_body(&[URL, "https://images.example/dog.png"]);
        body["stream"] = json!(stream);
        let (status, text) = post(&app, ENDPOINTS[0], &body).await;
        assert_eq!(status, StatusCode::GATEWAY_TIMEOUT, "{stream}: {text}");
        let json = parse(&text);
        assert_eq!(json["error"]["code"], "request_timeout", "{json}");
        assert_eq!(json["error"]["phase"], "image_fetch", "{json}");
        let message = json["error"]["message"].as_str().unwrap_or_default();
        assert!(
            message.starts_with("request exceeded the server's per-request timeout of 300 ms"),
            "{message}"
        );
        assert!(
            message.contains("while fetching one of the request's remote images"),
            "{message}"
        );
        assert!(message.contains("per-image deadline"), "{message}");

        tokio::time::sleep(Duration::from_millis(1_800)).await;
        assert_eq!(stub.calls(), 1, "the second image was never fetched");
    }
}

/// A deadline with room for the fetch changes nothing: the same answer as
/// with no deadline.
#[tokio::test]
async fn a_generous_deadline_changes_nothing_for_a_fetched_image() {
    let stub = StubFetcher::serving(png(), Duration::from_millis(20));
    let with_deadline = parse(
        &post(
            &router(fetching(stub.clone()), 600_000),
            ENDPOINTS[0],
            &image_body(&[URL]),
        )
        .await
        .1,
    );
    let without = parse(
        &post(
            &router(fetching(stub), 0),
            ENDPOINTS[0],
            &image_body(&[URL]),
        )
        .await
        .1,
    );
    assert_eq!(with_deadline["choices"], without["choices"]);
    assert_eq!(with_deadline["usage"], without["usage"]);
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. A client that goes away
// ─────────────────────────────────────────────────────────────────────────────

/// A client that disconnects while its first image is being fetched (the
/// handler's future is dropped): the fetch in flight ends by itself, and no
/// further image of that request is fetched — on either endpoint.
#[tokio::test]
async fn a_client_that_goes_away_mid_fetch_leaves_no_fetch_behind() {
    for path in ENDPOINTS {
        let stub = StubFetcher::serving(png(), Duration::from_millis(800));
        let app = router(fetching(stub.clone()), 0);
        let body = image_body(&[
            URL,
            "https://images.example/dog.png",
            "https://images.example/owl.png",
        ]);
        let abandoned = tokio::time::timeout(
            Duration::from_millis(200),
            app.clone().oneshot(request(path, &body)),
        )
        .await;
        assert!(
            abandoned.is_err(),
            "{path}: still fetching when the client left"
        );
        tokio::time::sleep(Duration::from_millis(1_500)).await;
        assert_eq!(stub.calls(), 1, "{path}: nothing after the fetch in flight");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. One request's fetches: one after another, and never past the cap
// ─────────────────────────────────────────────────────────────────────────────

/// The images of one request are fetched one after another — never two at
/// once — and each exactly once, on either endpoint.
#[tokio::test]
async fn one_requests_images_are_fetched_one_after_another() {
    let urls = [
        URL,
        "https://images.example/dog.png",
        "https://images.example/owl.png",
    ];
    for path in ENDPOINTS {
        let stub = StubFetcher::serving(png(), Duration::from_millis(40));
        let app = router(fetching(stub.clone()), 0);
        let (status, text) = post(&app, path, &image_body(&urls)).await;
        assert_eq!(status, StatusCode::OK, "{path}: {text}");
        assert_eq!(stub.calls(), urls.len(), "{path}: one fetch per image");
        assert_eq!(stub.most_in_flight(), 1, "{path}: one fetch at a time");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. A fetch in flight learns its request was abandoned; a fetcher at
//    capacity is a retryable 503
// ─────────────────────────────────────────────────────────────────────────────

/// Waits, like a real fetcher's transfer, until the request it fetches for
/// is abandoned (bounded), recording whether it could see the request and
/// how long the abandonment took to reach it.
#[derive(Default)]
struct Watching {
    calls: AtomicUsize,
    saw_request: AtomicUsize,
    /// Set once the fetch saw its request abandoned.
    saw_abandoned: std::sync::Mutex<Option<std::time::Instant>>,
}

impl Watching {
    fn abandoned_at(&self) -> Option<std::time::Instant> {
        self.saw_abandoned.lock().ok().and_then(|seen| *seen)
    }
}

impl RemoteImageFetcher for Watching {
    fn fetch(&self, url: &str, _max_bytes: usize) -> Result<Vec<u8>, ImageInputError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let Some(request) = CurrentFetch::on_this_thread() else {
            return Err(RemoteFetchFailure::Unavailable {
                reason: "no request".to_string(),
            }
            .into_error(url));
        };
        self.saw_request.fetch_add(1, Ordering::SeqCst);
        let start = std::time::Instant::now();
        while start.elapsed() < Duration::from_secs(20) {
            if request.is_abandoned() {
                if let Ok(mut seen) = self.saw_abandoned.lock() {
                    *seen = Some(std::time::Instant::now());
                }
                return Err(RemoteFetchFailure::Abandoned.into_error(url));
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        Err(RemoteFetchFailure::TimedOut { ms: 20_000 }.into_error(url))
    }
}

/// Wait (bounded) until `watching` saw its request abandoned; when it did.
async fn abandonment_seen(watching: &Watching) -> Option<std::time::Instant> {
    let start = std::time::Instant::now();
    while start.elapsed() < Duration::from_secs(10) {
        if let Some(at) = watching.abandoned_at() {
            return Some(at);
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    watching.abandoned_at()
}

/// A client that goes away mid-fetch, on either endpoint: the fetch in
/// flight learns it at once (a fetcher can drop it there), and no further
/// image is fetched.
#[tokio::test]
async fn a_fetch_in_flight_learns_that_its_client_went_away() {
    for path in ENDPOINTS {
        let watching = Arc::new(Watching::default());
        let app = router(fetching(watching.clone()), 0);
        let body = image_body(&[URL, "https://images.example/dog.png"]);
        let client = tokio::spawn(app.clone().oneshot(request(path, &body)));
        let start = std::time::Instant::now();
        while watching.saw_request.load(Ordering::SeqCst) == 0
            && start.elapsed() < Duration::from_secs(10)
        {
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        assert_eq!(
            watching.saw_request.load(Ordering::SeqCst),
            1,
            "{path}: the fetcher sees the request it fetches for"
        );
        client.abort();
        let left_at = std::time::Instant::now();
        assert!(client.await.is_err(), "{path}: still fetching");
        let seen = abandonment_seen(&watching)
            .await
            .unwrap_or_else(|| panic!("{path}: the fetch never learned its client went away"));
        assert!(
            seen.saturating_duration_since(left_at) < Duration::from_secs(2),
            "{path}: the abandonment reached the fetch in flight promptly"
        );
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(
            watching.calls.load(Ordering::SeqCst),
            1,
            "{path}: no further image is fetched"
        );
    }
}

/// The request's deadline expires mid-fetch, on either endpoint: a `504`
/// naming `image_fetch`, and the fetch in flight learns its request is gone.
#[tokio::test]
async fn a_fetch_in_flight_learns_that_its_deadline_expired() {
    for path in ENDPOINTS {
        let watching = Arc::new(Watching::default());
        let app = router(fetching(watching.clone()), 300);
        let (status, text) = post(&app, path, &image_body(&[URL, URL])).await;
        let answered_at = std::time::Instant::now();
        assert_eq!(status, StatusCode::GATEWAY_TIMEOUT, "{path}: {text}");
        let json = parse(&text);
        assert_eq!(json["error"]["code"], "request_timeout", "{path}: {json}");
        assert_eq!(json["error"]["phase"], "image_fetch", "{path}: {json}");
        let seen = abandonment_seen(&watching)
            .await
            .unwrap_or_else(|| panic!("{path}: the fetch never learned its deadline expired"));
        assert!(
            seen.saturating_duration_since(answered_at) < Duration::from_secs(2),
            "{path}: the deadline reached the fetch in flight promptly"
        );
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(
            watching.calls.load(Ordering::SeqCst),
            1,
            "{path}: the second image is never fetched"
        );
    }
}

/// `/v1/chat/completions/extended` names the `image_fetch` stage too: its
/// deadline behaves exactly like the base endpoint's, streamed or not.
#[tokio::test]
async fn the_extended_endpoint_names_image_fetch_in_its_504() {
    for stream in [false, true] {
        let stub = StubFetcher::serving(png(), Duration::from_millis(1_200));
        let app = router(fetching(stub.clone()), 300);
        let mut body = image_body(&[URL, "https://images.example/dog.png"]);
        body["stream"] = json!(stream);
        let (status, text) = post(&app, ENDPOINTS[1], &body).await;
        assert_eq!(status, StatusCode::GATEWAY_TIMEOUT, "{stream}: {text}");
        let json = parse(&text);
        assert_eq!(json["error"]["code"], "request_timeout", "{json}");
        assert_eq!(json["error"]["type"], "server_error", "{json}");
        assert_eq!(json["error"]["phase"], "image_fetch", "{json}");
        let message = json["error"]["message"].as_str().unwrap_or_default();
        assert!(
            message.contains("while fetching one of the request's remote images"),
            "{message}"
        );
        tokio::time::sleep(Duration::from_millis(1_800)).await;
        assert_eq!(stub.calls(), 1, "the second image was never fetched");
    }
}

/// Refuses every fetch the way a fetcher at capacity does: reports the
/// overload to the request, then fails without opening anything.
struct AtCapacity {
    calls: AtomicUsize,
}

impl RemoteImageFetcher for AtCapacity {
    fn fetch(&self, url: &str, _max_bytes: usize) -> Result<Vec<u8>, ImageInputError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if let Some(request) = CurrentFetch::on_this_thread() {
            request.report_overloaded();
        }
        Err(RemoteFetchFailure::Unavailable {
            reason: "it is at capacity (8 fetches in flight and 8 waiting)".to_string(),
        }
        .into_error(url))
    }
}

/// A fetcher at capacity sheds the request: `503` with `Retry-After`,
/// `type: overloaded_error`, `code: image_fetch_overloaded`, on both
/// endpoints, streamed or not (before any stream opens). An overload is not
/// a fault in the request: never a `400`.
#[tokio::test]
async fn a_fetcher_at_capacity_is_a_retryable_503() {
    let fetcher = Arc::new(AtCapacity {
        calls: AtomicUsize::new(0),
    });
    let app = router(fetching(fetcher.clone()), 0);
    for path in ENDPOINTS {
        for stream in [false, true] {
            let mut body = image_body(&[URL]);
            body["stream"] = json!(stream);
            let response = app
                .clone()
                .oneshot(request(path, &body))
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
            assert!(message.starts_with("image 0:"), "{message}");
            assert!(message.contains("at capacity"), "{message}");
        }
    }
    assert_eq!(fetcher.calls.load(Ordering::SeqCst), 4);

    // A fetcher failure that is not an overload stays the request's `400`.
    let failing: Arc<dyn RemoteImageFetcher> = Arc::new(Failing);
    let (status, text) = post(
        &router(fetching(failing), 0),
        ENDPOINTS[0],
        &image_body(&[URL]),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{text}");
}

/// A request with more images than the per-request cap is refused as
/// `too_many_images` before any of them is fetched.
#[tokio::test]
async fn more_images_than_the_cap_are_refused_before_any_fetch() {
    let urls: Vec<String> = (0..=MAX_IMAGES_PER_REQUEST)
        .map(|i| format!("https://images.example/{i}.png"))
        .collect();
    let urls: Vec<&str> = urls.iter().map(String::as_str).collect();
    for path in ENDPOINTS {
        let stub = StubFetcher::serving(png(), Duration::ZERO);
        let app = router(fetching(stub.clone()), 0);
        let (status, text) = post(&app, path, &image_body(&urls)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{path}: {text}");
        let json = parse(&text);
        assert_eq!(json["error"]["code"], "too_many_images", "{json}");
        assert_eq!(stub.calls(), 0, "{path}: nothing was fetched");
    }
}
