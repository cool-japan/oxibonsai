//! The shipped `oxibonsai` CLI with a Bonsai 2 vision projector, end to end
//! (SV-11; bonsai2-design.md §6.2): the image path a user actually runs, as a
//! subprocess, on the binary the release ships.
//!
//! # Real-model gate (`bonsai2-vision-metal` capability)
//!
//! `real_27b_cli_image_turns_stay_on_metal_and_answer_the_golden_bonsai2`,
//! on the real Bonsai 2 27B `PQ2_0` and its projector, with prompt 1 of the
//! vision golden (the 256 x 192 fixture image, "Describe this image
//! briefly.", greedy, 32 tokens, thinking off):
//!
//! 1. `oxibonsai run --mmproj --image --backend auto` stays on the Metal
//!    hybrid runner — its own engine summary names the runner, the projector
//!    is loaded on the Metal tower, and the line `auto` would log were it to
//!    rebuild the engine on the CPU model is absent from a log that does
//!    carry `INFO` lines — and prints the fork's golden answer byte for
//!    byte, from a 67-row prompt (48 of them image rows);
//! 2. the same with `--backend cpu` prints the same answer on the CPU model
//!    and the CPU tower; the wall time of both runs is printed (the CPU
//!    model decodes the 27B an order of magnitude slower, so a downgrade
//!    that still fired would show here);
//! 3. `oxibonsai serve --backend auto --mmproj` answers a
//!    `/v1/chat/completions` request carrying the image as a data URI with
//!    the golden text, 67 prompt tokens and 32 completion tokens, and
//!    `/v1/models` lists the model under its file stem (the 27B files carry
//!    the placeholder `general.name` "Hf");
//! 4. a dense model with an image is refused with the typed
//!    `NOT_A_HYBRID_MODEL` before any image is decoded: `run --mmproj
//!    --image` exits naming it, `serve --mmproj` refuses to start naming
//!    it, and a text-only `serve` answers an image request with `400
//!    NOT_A_HYBRID_MODEL` — the image a deliberately corrupt PNG, whose
//!    decode would have answered with an `image_*` code instead.
//!
//! The binary is the one `oxibonsai_testkit::cli_bin` resolves (the release
//! gate's stage-0 build, `OXIBONSAI_CLI_BIN`, else one `--all-features`
//! release build), resolved only once the files are found. The files come
//! only from `$OXI_BONSAI2_PQ2_GGUF` and `$OXI_BONSAI2_MMPROJ_GGUF` (or the
//! release names under `$OXIBONSAI_MODELS_DIR`); the golden from
//! `$OXI_BONSAI2_VISION_GOLDEN_DIR`, else the model crate's vendored copy.
//! Absent files (or no Metal device) skip with an `executed: false` record —
//! a failure under `OXI_REQUIRE_MODEL_FILES=1`; `executed: true` (timed) is
//! written only after every leg passed. One real-model process runs at a
//! time.
//!
//! # Hermetic case (every run)
//!
//! `a_dense_model_with_an_image_is_refused_before_any_decode` drives step 4
//! on the test kit's tiny dense model and synthetic projector, with the
//! binary cargo builds for this test.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

use oxibonsai_testkit::capability::{record_skipped, record_timed, Capability};

const PQ2_ENV: &str = "OXI_BONSAI2_PQ2_GGUF";
const MMPROJ_ENV: &str = "OXI_BONSAI2_MMPROJ_GGUF";
const GOLDEN_ENV: &str = "OXI_BONSAI2_VISION_GOLDEN_DIR";
const MODELS_DIR_ENV: &str = "OXIBONSAI_MODELS_DIR";
const REQUIRE_ENV: &str = "OXI_REQUIRE_MODEL_FILES";
const PQ2_FILE: &str = "Ternary-Bonsai-2-27B-PQ2_0.gguf";
const MMPROJ_FILE: &str = "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf";
/// The capability the real gate records under.
const CAPABILITY: Capability = Capability::Bonsai2VisionMetal;
const REAL_TEST: &str = "oxibonsai-cli::bonsai2_vision_cli_tests::\
                         real_27b_cli_image_turns_stay_on_metal_and_answer_the_golden_bonsai2";
/// Prompt 1 of the vision golden.
const PROMPT_TEXT: &str = "Describe this image briefly.";
/// The golden's prompt rows: 19 text tokens around 48 image rows.
const GOLDEN_PROMPT_TOKENS: u64 = 67;
/// The golden's answer length.
const GOLDEN_COMPLETION_TOKENS: u64 = 32;
/// The engine summary's kernel label of a Metal-backed hybrid engine.
const METAL_RUNNER_LABEL: &str = "Metal (hybrid runner)";
/// The engine summary's tier reason of a hybrid engine on the Metal runner.
const METAL_RUNNER_REASON: &str = "decodes on the Metal hybrid runner";
/// The engine summary's tier reason of a hybrid engine on the CPU model.
const CPU_MODEL_REASON: &str = "runs on the CPU model";
/// What `--backend auto` logs when it rebuilds a vision engine on the CPU
/// model — the line whose absence proves the run stayed where `auto` put it.
const REBUILD_LINE: &str = "using the CPU engine for this session";
/// A PNG signature followed by bytes no decoder accepts: decoding it fails
/// with an `image_*` code, so a refusal that names anything else came first.
const CORRUPT_PNG: &[u8] = b"\x89PNG\r\n\x1a\nthis is not an image";

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var(name)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .map(PathBuf::from)
}

/// A release file from its variable or `$OXIBONSAI_MODELS_DIR` only — never
/// a workspace path, so an ordinary test run maps no 27B.
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

/// The vision golden: the fork's answers and the image they describe.
fn golden_dir() -> PathBuf {
    env_path(GOLDEN_ENV).unwrap_or_else(|| {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("crates/oxibonsai-model/tests/fixtures/bonsai2_golden_vision")
    })
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

/// A data URI of `png`.
#[cfg(feature = "server")]
fn png_data_uri(png: &[u8]) -> String {
    format!("data:image/png;base64,{}", base64(png))
}

/// An `oxibonsai` invocation with an environment that cannot change what it
/// does: `INFO` logging on (so the absence of a log line means something),
/// and no inherited model, kernel tier, media directory, pool size or
/// capability manifest.
fn cli(binary: &Path) -> Command {
    let mut command = Command::new(binary);
    command
        .env("RUST_LOG", "info")
        .env_remove("OXIBONSAI_KERNEL_TIER")
        .env_remove("OXI_MODEL")
        .env_remove("OXI_TOKENIZER")
        .env_remove("OXI_MEDIA_PATH")
        .env_remove("OXI_ALLOW_IMAGE_URL_FETCH")
        .env_remove("OXIBONSAI_ENGINE_POOL_SIZE")
        .env_remove("OXIBONSAI_CAPABILITY_REPORT")
        .stdin(Stdio::null());
    command
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// `log` without its ANSI colour sequences (`ESC [ ... m`), so a log line's
/// level reads as the plain word the formatter prints between them.
fn strip_ansi(log: &str) -> String {
    let mut out = String::with_capacity(log.len());
    let mut chars = log.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' && chars.peek() == Some(&'[') {
            for next in chars.by_ref() {
                if next.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// The standard error of a finished command, colour sequences removed.
fn log_text(bytes: &[u8]) -> String {
    strip_ansi(&text(bytes))
}

/// The last `lines` lines of `log`, for a failure message.
fn tail(log: &str, lines: usize) -> String {
    let all: Vec<&str> = log.lines().collect();
    all[all.len().saturating_sub(lines)..].join("\n")
}

/// A free loopback port: bound, read and released.
#[cfg(feature = "server")]
fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .and_then(|listener| listener.local_addr())
        .map(|addr| addr.port())
        .expect("a free loopback port")
}

/// A `serve` child: killed and reaped when dropped, so a failed assertion
/// never leaves a server (and a mapped 27B) behind. Its output goes to a
/// file — never an undrained pipe, which a busy log would fill and stall the
/// server on.
#[cfg(feature = "server")]
struct ServeChild {
    child: std::process::Child,
    log: PathBuf,
    base: String,
}

#[cfg(feature = "server")]
impl ServeChild {
    /// Start `oxibonsai serve` on a free loopback port with `args`.
    fn start(binary: &Path, args: &[&str], tag: &str) -> Self {
        let port = free_port();
        let log =
            oxibonsai_testkit::temp_path::unique_path(&format!("cli_vision_serve_{tag}"), ".log");
        let file = std::fs::File::create(&log).expect("the server log file");
        let file_err = file.try_clone().expect("a second handle on the log");
        let port_text = port.to_string();
        let mut command = cli(binary);
        command
            .arg("serve")
            .args(args)
            .args(["--host", "127.0.0.1", "--port", port_text.as_str()])
            .stdout(Stdio::from(file))
            .stderr(Stdio::from(file_err));
        let child = command.spawn().expect("spawn oxibonsai serve");
        Self {
            child,
            log,
            base: format!("http://127.0.0.1:{port}"),
        }
    }

    fn log_text(&self) -> String {
        strip_ansi(&std::fs::read_to_string(&self.log).unwrap_or_default())
    }

    /// Wait until `/health` answers `200`, at most `limit`; fails with the
    /// log's tail if the server exits first or never answers.
    fn wait_ready(&mut self, client: &reqwest::blocking::Client, limit: Duration) {
        let started = Instant::now();
        loop {
            if let Ok(Some(status)) = self.child.try_wait() {
                panic!(
                    "oxibonsai serve exited ({status}) before it served:\n{}",
                    tail(&self.log_text(), 40)
                );
            }
            if let Ok(response) = client.get(format!("{}/health", self.base)).send() {
                if response.status().is_success() {
                    return;
                }
            }
            assert!(
                started.elapsed() < limit,
                "oxibonsai serve did not answer /health within {limit:?}:\n{}",
                tail(&self.log_text(), 40)
            );
            std::thread::sleep(Duration::from_millis(250));
        }
    }

    /// Wait for the process to exit on its own (a refused start), at most
    /// `limit`; the exit status and the log.
    fn wait_exit(mut self, limit: Duration) -> (std::process::ExitStatus, String) {
        let started = Instant::now();
        loop {
            if let Ok(Some(status)) = self.child.try_wait() {
                return (status, self.log_text());
            }
            assert!(
                started.elapsed() < limit,
                "oxibonsai serve kept running for {limit:?} although it should have refused to \
                 start:\n{}",
                tail(&self.log_text(), 40)
            );
            std::thread::sleep(Duration::from_millis(100));
        }
    }
}

#[cfg(feature = "server")]
impl Drop for ServeChild {
    fn drop(&mut self) {
        // Already exited is fine; either way it is reaped here.
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_file(&self.log);
    }
}

/// A blocking client patient enough for a 27B image turn.
#[cfg(feature = "server")]
fn http_client() -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()
        .expect("an HTTP client")
}

/// The golden chat request: one user turn of an image part and the prompt
/// text, greedy, 32 tokens, thinking off.
#[cfg(feature = "server")]
fn image_request(image_url: &str) -> serde_json::Value {
    serde_json::json!({
        "messages": [{
            "role": "user",
            "content": [
                { "type": "image_url", "image_url": { "url": image_url } },
                { "type": "text", "text": PROMPT_TEXT },
            ],
        }],
        "max_tokens": GOLDEN_COMPLETION_TOKENS,
        "temperature": 0.0,
        "chat_template_kwargs": { "enable_thinking": false },
    })
}

/// POST `body` as JSON; the status and the parsed body.
#[cfg(feature = "server")]
fn post_json(
    client: &reqwest::blocking::Client,
    url: &str,
    body: &serde_json::Value,
) -> (reqwest::StatusCode, serde_json::Value) {
    let response = client
        .post(url)
        .json(body)
        .send()
        .unwrap_or_else(|e| panic!("POST {url}: {e}"));
    let status = response.status();
    let text = response.text().unwrap_or_default();
    let json =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("{url}: not JSON ({e}): {text}"));
    (status, json)
}

/// Step 4 on `binary`: a dense model with an image is refused with
/// `NOT_A_HYBRID_MODEL` before any image is decoded — by `run` (exit, typed
/// line), by `serve --mmproj` (refused start) and by a text-only `serve`
/// (`400` over HTTP). `projector` is any valid projector GGUF.
fn assert_dense_model_refuses_images(binary: &Path, projector: &Path, label: &str) {
    let dense = oxibonsai_testkit::temp_path::write_temp_file(
        "cli_vision_dense",
        ".gguf",
        &oxibonsai_testkit::gguf_fixture::tiny_dense_qwen3_gguf(7).expect("the tiny dense model"),
    )
    .expect("write the tiny dense model");
    let corrupt =
        oxibonsai_testkit::temp_path::write_temp_file("cli_vision_corrupt", ".png", CORRUPT_PNG)
            .expect("write the corrupt image");
    let dense_arg = dense.to_string_lossy().into_owned();
    let projector_arg = projector.to_string_lossy().into_owned();

    // `run --mmproj --image` on a dense model.
    let out = cli(binary)
        .args([
            "run",
            "--model",
            dense_arg.as_str(),
            "--mmproj",
            projector_arg.as_str(),
        ])
        .args([
            "--image",
            corrupt.to_string_lossy().as_ref(),
            "--prompt",
            PROMPT_TEXT,
        ])
        .args(["--max-tokens", "4", "--temperature", "0"])
        .output()
        .expect("run oxibonsai run");
    let stderr = log_text(&out.stderr);
    assert!(
        !out.status.success(),
        "{label}: a dense model must refuse the image: {stderr}"
    );
    assert!(
        stderr.contains("[NOT_A_HYBRID_MODEL]"),
        "{label}: the typed refusal: {stderr}"
    );
    assert!(
        !stderr.contains("[image_"),
        "{label}: refused before the image was decoded: {stderr}"
    );
    report(&format!(
        "{label}: run on a dense model refused: {}",
        tail(&stderr, 1)
    ));

    #[cfg(feature = "server")]
    {
        // `serve --mmproj` on a dense model refuses to start, typed.
        let refused = ServeChild::start(
            binary,
            &[
                "--model",
                dense_arg.as_str(),
                "--mmproj",
                projector_arg.as_str(),
            ],
            "dense_mmproj",
        );
        let (status, log) = refused.wait_exit(Duration::from_secs(120));
        assert!(
            !status.success(),
            "{label}: serve --mmproj on a dense model: {log}"
        );
        assert!(
            log.contains("[NOT_A_HYBRID_MODEL]"),
            "{label}: serve names the refusal: {}",
            tail(&log, 20)
        );

        // A text-only server on a dense model: an image request is the
        // engine's typed 400, before any decode.
        let client = http_client();
        let mut server = ServeChild::start(binary, &["--model", dense_arg.as_str()], "dense_text");
        server.wait_ready(&client, Duration::from_secs(120));
        let (status, json) = post_json(
            &client,
            &format!("{}/v1/chat/completions", server.base),
            &image_request(&png_data_uri(CORRUPT_PNG)),
        );
        assert_eq!(status, reqwest::StatusCode::BAD_REQUEST, "{label}: {json}");
        assert_eq!(
            json["error"]["code"], "NOT_A_HYBRID_MODEL",
            "{label}: {json}"
        );
        report(&format!(
            "{label}: serve on a dense model answered the image request with {} {}",
            status.as_u16(),
            json["error"]["code"]
        ));
    }
}

/// The hermetic form of step 4: the test kit's tiny dense model and
/// synthetic projector, through the binary cargo builds for this test.
#[test]
fn a_dense_model_with_an_image_is_refused_before_any_decode() {
    let spec = oxibonsai_testkit::mmproj_fixture::MmprojFixtureSpec::tiny();
    let projector = oxibonsai_testkit::temp_path::write_temp_file(
        "cli_vision_projector",
        ".gguf",
        &oxibonsai_testkit::mmproj_fixture::synthetic_mmproj_gguf(&spec).expect("the projector"),
    )
    .expect("write the projector");
    assert_dense_model_refuses_images(
        Path::new(env!("CARGO_BIN_EXE_oxibonsai")),
        &projector,
        "synthetic",
    );
    let _ = std::fs::remove_file(&projector);
}

/// One `run` of the golden prompt; the output and its wall time.
fn run_golden(
    binary: &Path,
    model: &Path,
    mmproj: &Path,
    image: &Path,
    backend: &str,
) -> (Output, Duration) {
    let started = Instant::now();
    let output = cli(binary)
        .arg("run")
        .args(["--model", model.to_string_lossy().as_ref()])
        .args(["--mmproj", mmproj.to_string_lossy().as_ref()])
        .args(["--image", image.to_string_lossy().as_ref()])
        .args(["--prompt", PROMPT_TEXT, "--no-think"])
        .args(["--temperature", "0", "--max-tokens", "32"])
        .args(["--backend", backend])
        .output()
        .expect("run oxibonsai run");
    (output, started.elapsed())
}

/// The engine summary line `run` prints once the engine is built.
fn engine_summary(stderr: &str) -> String {
    stderr
        .lines()
        .find(|line| line.starts_with("Resolved model:"))
        .unwrap_or_else(|| panic!("no engine summary in:\n{}", tail(stderr, 40)))
        .to_string()
}

/// Steps 1-4 on the real files (see the module docs).
fn run_real_gate(binary: &Path, model: &Path, mmproj: &Path) {
    let started = Instant::now();
    let golden: serde_json::Value = serde_json::from_slice(
        &std::fs::read(golden_dir().join("vision.prompt1.cpu.json")).expect("the golden"),
    )
    .expect("golden JSON");
    let golden_text = golden["response"]["choices"][0]["message"]["content"]
        .as_str()
        .expect("golden content")
        .to_string();
    assert_eq!(
        golden["response"]["usage"]["prompt_tokens"].as_u64(),
        Some(GOLDEN_PROMPT_TOKENS),
        "the golden's own usage"
    );
    let image = golden_dir().join("fixture_256x192.png");

    // 1. `--backend auto` on this Metal host: the Metal runner, the golden.
    let (auto, auto_wall) = run_golden(binary, model, mmproj, &image, "auto");
    let auto_stderr = log_text(&auto.stderr);
    assert!(
        auto.status.success(),
        "run --backend auto failed:\n{}",
        tail(&auto_stderr, 40)
    );
    let summary = engine_summary(&auto_stderr);
    report(&format!("cli run --backend auto: {summary}"));
    assert!(
        summary.contains(METAL_RUNNER_LABEL) && summary.contains(METAL_RUNNER_REASON),
        "auto stays on the Metal runner: {summary}"
    );
    assert!(
        auto_stderr.contains(" INFO "),
        "INFO logging reached the run, so a missing line means something:\n{}",
        tail(&auto_stderr, 20)
    );
    assert!(
        !auto_stderr.contains(REBUILD_LINE),
        "auto rebuilt the engine on the CPU model:\n{}",
        tail(&auto_stderr, 40)
    );
    assert!(
        auto_stderr.contains("vision projector loaded on the metal tower"),
        "the Metal tower beside the Metal runner:\n{}",
        tail(&auto_stderr, 40)
    );
    assert!(
        auto_stderr.contains(&format!(
            "{GOLDEN_PROMPT_TOKENS} prompt + {GOLDEN_COMPLETION_TOKENS} generated"
        )),
        "the 67-row image prompt and 32 tokens:\n{}",
        tail(&auto_stderr, 10)
    );
    let auto_answer = text(&auto.stdout);
    report(&format!("cli run --backend auto answer: {auto_answer:?}"));
    assert_eq!(
        auto_answer, golden_text,
        "the golden answer, byte for byte, on Metal"
    );

    // 2. `--backend cpu`: the same answer on the CPU model and tower.
    let (cpu, cpu_wall) = run_golden(binary, model, mmproj, &image, "cpu");
    let cpu_stderr = log_text(&cpu.stderr);
    assert!(
        cpu.status.success(),
        "run --backend cpu failed:\n{}",
        tail(&cpu_stderr, 40)
    );
    let cpu_summary = engine_summary(&cpu_stderr);
    report(&format!("cli run --backend cpu: {cpu_summary}"));
    assert!(
        cpu_summary.contains(CPU_MODEL_REASON) && !cpu_summary.contains(METAL_RUNNER_REASON),
        "--backend cpu decodes on the CPU model: {cpu_summary}"
    );
    assert!(
        cpu_stderr.contains("vision projector loaded on the cpu tower"),
        "the CPU tower beside the CPU model:\n{}",
        tail(&cpu_stderr, 40)
    );
    assert!(
        cpu_stderr.contains(&format!(
            "{GOLDEN_PROMPT_TOKENS} prompt + {GOLDEN_COMPLETION_TOKENS} generated"
        )),
        "{}",
        tail(&cpu_stderr, 10)
    );
    assert_eq!(
        text(&cpu.stdout),
        golden_text,
        "the golden answer on the CPU"
    );
    report(&format!(
        "cli wall time: run --backend auto (Metal runner) {:.1} s, run --backend cpu (CPU model) \
         {:.1} s",
        auto_wall.as_secs_f64(),
        cpu_wall.as_secs_f64()
    ));

    // 3. `serve --backend auto --mmproj`: the golden over HTTP, the file stem
    //    as the model id.
    #[cfg(feature = "server")]
    {
        let png = std::fs::read(&image).expect("the fixture image");
        let client = http_client();
        let serve_started = Instant::now();
        let mut server = ServeChild::start(
            binary,
            &[
                "--model",
                model.to_string_lossy().as_ref(),
                "--mmproj",
                mmproj.to_string_lossy().as_ref(),
                "--backend",
                "auto",
                "--request-timeout-ms",
                "600000",
            ],
            "real",
        );
        server.wait_ready(&client, Duration::from_secs(900));
        let log = server.log_text();
        assert!(
            log.contains(METAL_RUNNER_LABEL) && log.contains(METAL_RUNNER_REASON),
            "the server's replicas decode on the Metal runner:\n{}",
            tail(&log, 40)
        );
        assert!(
            log.contains("vision projector loaded on the metal tower"),
            "{}",
            tail(&log, 40)
        );
        assert!(!log.contains(REBUILD_LINE), "{}", tail(&log, 40));
        let models: serde_json::Value = client
            .get(format!("{}/v1/models", server.base))
            .send()
            .and_then(reqwest::blocking::Response::json)
            .expect("GET /v1/models");
        let stem = model
            .file_stem()
            .and_then(std::ffi::OsStr::to_str)
            .expect("a file stem");
        assert_eq!(models["data"][0]["id"], stem, "{models}");
        let request_started = Instant::now();
        let (status, json) = post_json(
            &client,
            &format!("{}/v1/chat/completions", server.base),
            &image_request(&png_data_uri(&png)),
        );
        assert_eq!(status, reqwest::StatusCode::OK, "{json}");
        report(&format!(
            "cli serve --backend auto: prompt_tokens {} completion_tokens {} in {:.1} s (server \
             up after {:.1} s)",
            json["usage"]["prompt_tokens"],
            json["usage"]["completion_tokens"],
            request_started.elapsed().as_secs_f64(),
            (request_started - serve_started).as_secs_f64()
        ));
        assert_eq!(
            json["usage"]["prompt_tokens"], GOLDEN_PROMPT_TOKENS,
            "{json}"
        );
        assert_eq!(
            json["usage"]["completion_tokens"], GOLDEN_COMPLETION_TOKENS,
            "{json}"
        );
        assert_eq!(json["model"], stem, "{json}");
        assert_eq!(
            json["choices"][0]["message"]["content"], golden_text,
            "the golden answer over HTTP"
        );
    }

    // 4. A dense model with an image: typed, before any decode.
    assert_dense_model_refuses_images(binary, mmproj, "real projector");

    record_timed(CAPABILITY, true, REAL_TEST, started.elapsed());
    report(&format!(
        "CAPABILITY-REPORT capability={CAPABILITY} executed=true test={REAL_TEST} duration_ms={}",
        started.elapsed().as_millis()
    ));
}

/// Whether this build and host have a Metal device for the runner.
fn metal_available() -> bool {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        match oxibonsai_kernels::MetalGraph::shared_device() {
            Ok(_) => true,
            Err(oxibonsai_kernels::MetalGraphError::DeviceNotFound) => false,
            Err(e) => panic!("the Metal device must open on this host: {e}"),
        }
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        false
    }
}

/// Record a skip (or, under `OXI_REQUIRE_MODEL_FILES=1`, fail) with `why`.
fn skip(why: &str) {
    assert!(
        !require_model_files(),
        "{REQUIRE_ENV}=1: {REAL_TEST}: {why}"
    );
    record_skipped(CAPABILITY, REAL_TEST);
    report(&format!(
        "CAPABILITY-REPORT capability={CAPABILITY} executed=false test={REAL_TEST} ({why})"
    ));
}

#[test]
fn real_27b_cli_image_turns_stay_on_metal_and_answer_the_golden_bonsai2() {
    let (Some(model), Some(mmproj)) = (locate(PQ2_ENV, PQ2_FILE), locate(MMPROJ_ENV, MMPROJ_FILE))
    else {
        skip(&format!(
            "set {PQ2_ENV} and {MMPROJ_ENV} (or {MODELS_DIR_ENV}) to {PQ2_FILE} and {MMPROJ_FILE}"
        ));
        return;
    };
    if !metal_available() {
        skip("this build or host has no Metal device for the hybrid runner");
        return;
    }
    // Resolved only now the files are found: a run without them never builds.
    let binary = oxibonsai_testkit::cli_bin::resolve_cli_binary()
        .unwrap_or_else(|e| panic!("the oxibonsai binary: {e}"));
    report(&format!("{REAL_TEST}: binary {}", binary.display()));
    run_real_gate(&binary, &model, &mmproj);
}

/// The log reader sees a coloured line's level as the plain word between
/// spaces, and leaves uncoloured text alone.
#[test]
fn strip_ansi_leaves_the_plain_log_line() {
    let coloured = "\u{1b}[2m2026-10-05T03:16:13Z\u{1b}[0m \u{1b}[32m INFO\u{1b}[0m \u{1b}[2mcrate\u{1b}[0m\u{1b}[2m:\u{1b}[0m message";
    assert_eq!(
        strip_ansi(coloured),
        "2026-10-05T03:16:13Z  INFO crate: message"
    );
    assert_eq!(strip_ansi("plain [image 1] text"), "plain [image 1] text");
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
