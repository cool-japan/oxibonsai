//! Image content through the OpenAI-compatible chat endpoint on the Metal
//! backend (SV-11; bonsai2-design.md §6.2): the real Bonsai 2 27B `PQ2_0` on
//! the Metal hybrid runner and its projector on the Metal vision tower,
//! behind the router built directly (no CLI).
//!
//! `real_27b_metal_engine_serves_the_golden_image_prompt_over_http_bonsai2`:
//!
//! * the engine is built with `--backend metal` inside a
//!   `HybridLoadScope` carrying the Metal tower's resident bytes
//!   (`VisionService::metal_footprint`), so its KV window leaves room for the
//!   tower, and the vision service is loaded for the engine
//!   (`VisionService::load_for_engine`: the engine and the projector checked
//!   together, the tower of the engine's own executor) — the Metal tower
//!   only, beside the runner as the process's second Metal session;
//! * prompt 1 of the vision golden as a data-URI `image_url` request
//!   (`enable_thinking = false`, greedy, 32 tokens) answers with the fork's
//!   golden text byte for byte, 67 prompt tokens (48 of them image rows, the
//!   fork server's own `usage.prompt_tokens`) and 32 completion tokens.
//!
//! The files come only from `$OXI_BONSAI2_PQ2_GGUF` and
//! `$OXI_BONSAI2_MMPROJ_GGUF` (or the release names under
//! `$OXIBONSAI_MODELS_DIR`); the golden from
//! `$OXI_BONSAI2_VISION_GOLDEN_DIR`, else the model crate's vendored copy.
//! Absent files (or no Metal device) skip with a `bonsai2-vision-metal` /
//! `executed: false` capability record — a failure under
//! `OXI_REQUIRE_MODEL_FILES=1`.

#![cfg(feature = "server")]

use std::io::Write as _;
use std::path::PathBuf;
use std::time::Duration;

use oxibonsai_testkit::capability::{record_timed, Capability};

const PQ2_ENV: &str = "OXI_BONSAI2_PQ2_GGUF";
const MMPROJ_ENV: &str = "OXI_BONSAI2_MMPROJ_GGUF";
const MODELS_DIR_ENV: &str = "OXIBONSAI_MODELS_DIR";
const REQUIRE_ENV: &str = "OXI_REQUIRE_MODEL_FILES";
const PQ2_FILE: &str = "Ternary-Bonsai-2-27B-PQ2_0.gguf";
const MMPROJ_FILE: &str = "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf";
const CAPABILITY: Capability = Capability::Bonsai2VisionMetal;
const REAL_TEST: &str = "oxibonsai-runtime::bonsai2_vision_metal_runtime_tests::\
                         real_27b_metal_engine_serves_the_golden_image_prompt_over_http_bonsai2";

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var(name)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .map(PathBuf::from)
}

/// A release file from its variable or `$OXIBONSAI_MODELS_DIR` only.
fn locate(env: &str, file: &str) -> Option<PathBuf> {
    env_path(env)
        .or_else(|| env_path(MODELS_DIR_ENV).map(|dir| dir.join(file)))
        .filter(|p| p.is_file())
}

fn require_model_files() -> bool {
    std::env::var(REQUIRE_ENV).is_ok_and(|v| v.trim() == "1")
}

/// A line on the process's standard error that libtest does not capture.
fn report(line: &str) {
    let mut stderr = std::io::stderr().lock();
    // Bookkeeping only: a failed diagnostic write must not fail the test.
    let _ = writeln!(stderr, "{line}");
}

/// Append one record to the capability manifest through the test kit (one
/// `write_all`, the documented schema); `duration_ms` is attached when the
/// gate measured one.
fn record(executed: bool, duration: Option<Duration>) {
    match duration {
        Some(d) => record_timed(CAPABILITY, executed, REAL_TEST, d),
        None => oxibonsai_testkit::capability::record(CAPABILITY, executed, REAL_TEST),
    }
    report(&format!(
        "CAPABILITY-REPORT capability={CAPABILITY} executed={executed} test={REAL_TEST}{}",
        duration.map_or_else(String::new, |d| format!(" duration_ms={}", d.as_millis()))
    ));
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

#[cfg(all(feature = "metal", target_os = "macos"))]
mod real {
    use super::*;

    use std::path::Path;
    use std::sync::Arc;
    use std::time::Instant;

    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use serde_json::{json, Value};
    use tower::ServiceExt;

    use oxibonsai_core::gguf::reader::{mmap_gguf_file, GgufFile};
    use oxibonsai_model::vision::{ImageSourcePolicy, DEFAULT_IMAGE_MAX_TOKENS};
    use oxibonsai_runtime::engine::InferenceEngine;
    use oxibonsai_runtime::engine_hybrid_gpu::{
        HybridBackend, HybridLoadOptions, HybridLoadScope, MetalSessionReport,
    };
    use oxibonsai_runtime::engine_seam::Backend;
    use oxibonsai_runtime::sampling::SamplingParams;
    use oxibonsai_runtime::server::create_router;
    use oxibonsai_runtime::vision_prefill::VisionService;
    use oxibonsai_tokenizer::chat_templates::ResolvedChatTemplate;

    const GOLDEN_ENV: &str = "OXI_BONSAI2_VISION_GOLDEN_DIR";
    /// The real engine's KV window: the 67-row prompt plus 32 tokens, with
    /// room.
    const REAL_KV_WINDOW: usize = 512;
    const PROMPT_TEXT: &str = "Describe this image briefly.";

    /// The vision golden: the fork's answers and the 256 x 192 image they
    /// describe.
    fn golden_dir() -> PathBuf {
        env_path(GOLDEN_ENV).unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../oxibonsai-model/tests/fixtures/bonsai2_golden_vision")
        })
    }

    async fn post(app: &axum::Router, path: &str, body: &Value) -> (StatusCode, Value) {
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
        let json = serde_json::from_slice(&bytes).unwrap_or_else(|e| {
            panic!(
                "{path}: not JSON ({e}): {}",
                String::from_utf8_lossy(&bytes)
            )
        });
        (status, json)
    }

    pub(super) fn metal_available() -> bool {
        match oxibonsai_kernels::MetalGraph::shared_device() {
            Ok(_) => true,
            Err(oxibonsai_kernels::MetalGraphError::DeviceNotFound) => false,
            Err(e) => panic!("the Metal device must open on this host: {e}"),
        }
    }

    pub(super) async fn run(model_path: &Path, mmproj_path: &Path) {
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
        let png = std::fs::read(golden_dir().join("fixture_256x192.png")).expect("the fixture");

        // What the Metal tower will keep resident, before any weight is
        // bound: the engine's KV window leaves room for it.
        let vision_bytes = VisionService::metal_footprint(mmproj_path, DEFAULT_IMAGE_MAX_TOKENS)
            .expect("the projector's Metal footprint");
        let mmap: &'static memmap2::Mmap = Box::leak(Box::new(
            mmap_gguf_file(model_path).expect("map the 27B GGUF"),
        ));
        let gguf: &'static GgufFile<'static> = Box::leak(Box::new(
            GgufFile::parse(mmap).expect("the 27B GGUF parses"),
        ));
        let loading = Instant::now();
        let engine = {
            let _hybrid = HybridLoadScope::enter(HybridLoadOptions {
                vision_resident_bytes: vision_bytes,
                prefill_chunk: None,
            });
            InferenceEngine::from_gguf_with_backend(
                gguf,
                SamplingParams {
                    temperature: 0.0,
                    ..SamplingParams::default()
                },
                42,
                REAL_KV_WINDOW,
                Backend::Metal,
            )
            .expect("the 27B loads on the Metal hybrid runner")
        };
        assert_eq!(engine.hybrid_backend(), Some(HybridBackend::Metal));
        let window = engine
            .hybrid_metal_window()
            .cloned()
            .expect("a Metal window");
        assert_eq!(window.vision_resident_bytes, vision_bytes);
        assert_eq!(window.window, REAL_KV_WINDOW);
        report(&format!(
            "{CAPABILITY} server: language model on Metal in {:.2}s; {}",
            loading.elapsed().as_secs_f64(),
            window.summary()
        ));
        let template = ResolvedChatTemplate::from_gguf(&gguf.metadata).expect("its chat template");
        let tok = oxibonsai_runtime::engine::tokenizer_from_gguf(gguf)
            .expect("its embedded tokenizer")
            .with_chat_template(template);

        let loading = Instant::now();
        let service = VisionService::load_for_engine(
            mmproj_path,
            DEFAULT_IMAGE_MAX_TOKENS,
            ImageSourcePolicy::server(None, false),
            &engine,
        )
        .expect("the projector loads for the Metal engine");
        assert!(
            service.tower().is_metal(),
            "a Metal engine holds the Metal tower only"
        );
        assert_eq!(
            service
                .check_engine(&engine)
                .expect("the projector fits the model"),
            HybridBackend::Metal
        );
        let sessions = MetalSessionReport::current().expect("a Metal build reports its sessions");
        assert!(
            sessions.live >= MetalSessionReport::sessions_for(true),
            "the runner's and the tower's sessions are live: {sessions:?}"
        );
        report(&format!(
            "{CAPABILITY} server: Metal sessions {}",
            sessions.describe(true)
        ));
        assert_eq!(service.tower().resident_bytes() as u64, vision_bytes);
        report(&format!(
            "{CAPABILITY} server: Metal vision tower loaded in {:.2}s, {vision_bytes} bytes \
             resident",
            loading.elapsed().as_secs_f64()
        ));
        let app = create_router(engine, Some(tok)).layer(axum::Extension(Arc::new(service)));

        let body = json!({
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "image_url", "image_url": {
                        "url": format!("data:image/png;base64,{}", base64(&png)) } },
                    { "type": "text", "text": PROMPT_TEXT },
                ],
            }],
            "max_tokens": 32,
            "temperature": 0.0,
            "chat_template_kwargs": { "enable_thinking": false },
        });
        let request_started = Instant::now();
        let (status, json) = post(&app, "/v1/chat/completions", &body).await;
        report(&format!(
            "{CAPABILITY} server: request (encode + prefill + 32-token decode) in {:.2}s",
            request_started.elapsed().as_secs_f64()
        ));
        assert_eq!(status, StatusCode::OK, "{json}");
        let content = json["choices"][0]["message"]["content"]
            .as_str()
            .expect("an answer")
            .to_string();
        report(&format!("{CAPABILITY} server answer: {content:?}"));
        report(&format!(
            "{CAPABILITY} server usage: prompt_tokens {} completion_tokens {}",
            json["usage"]["prompt_tokens"], json["usage"]["completion_tokens"]
        ));
        assert_eq!(json["usage"]["prompt_tokens"], json!(golden_prompt_tokens));
        assert_eq!(json["usage"]["completion_tokens"], json!(32));
        assert_eq!(json["choices"][0]["finish_reason"], json!("length"));
        assert_eq!(
            content, golden_text,
            "the fork's golden answer, byte for byte, on Metal"
        );
        record(true, Some(started.elapsed()));
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn real_27b_metal_engine_serves_the_golden_image_prompt_over_http_bonsai2() {
    let (Some(model_path), Some(mmproj_path)) =
        (locate(PQ2_ENV, PQ2_FILE), locate(MMPROJ_ENV, MMPROJ_FILE))
    else {
        assert!(
            !require_model_files(),
            "{REQUIRE_ENV}=1: {REAL_TEST} needs {PQ2_FILE} and {MMPROJ_FILE} (set {PQ2_ENV} / \
             {MMPROJ_ENV})"
        );
        report(&format!(
            "skip {REAL_TEST}: set {PQ2_ENV} and {MMPROJ_ENV} (or {MODELS_DIR_ENV})"
        ));
        record(false, None);
        return;
    };
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        if !real::metal_available() {
            assert!(
                !require_model_files(),
                "{REQUIRE_ENV}=1 but no Metal device"
            );
            record(false, None);
            return;
        }
        real::run(&model_path, &mmproj_path).await;
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        let _ = (model_path, mmproj_path);
        assert!(
            !require_model_files(),
            "{REQUIRE_ENV}=1 but this build has no Metal backend"
        );
        record(false, None);
    }
}

/// The request encoding helper against RFC 4648's own vectors.
#[test]
fn base64_matches_the_rfc_vectors() {
    for (input, want) in [
        ("", ""),
        ("f", "Zg=="),
        ("fo", "Zm8="),
        ("foo", "Zm9v"),
        ("foob", "Zm9vYg=="),
        ("fooba", "Zm9vYmE="),
        ("foobar", "Zm9vYmFy"),
    ] {
        assert_eq!(base64(input.as_bytes()), want, "{input:?}");
    }
}
