//! Remote image references through the real `oxibonsai` binary, up to the
//! point where the model would load: what `--help` says, which settings are
//! refused at start-up (naming the setting), and that validating an
//! `--image https://...` never opens a connection — without the opt-in it is
//! refused, with the opt-in a loopback literal is refused by the address
//! policy, and an allowlisted one passes validation unfetched (it is fetched
//! once, when the image is prepared after the model loads; the model here
//! does not exist, so the command stops before that).
//!
//! A text-only `run` / `chat` (no `--mmproj`) never reads the remote-image
//! variables: a stale or malformed one left in the shell or in `.env` cannot
//! stop it, while the same variables still stop a command that loads a
//! projector, before its model loads.
//!
//! Every listener counts the connections it accepts. The binary runs in a
//! temporary working directory (no `.env` above it) with the remote-image
//! variables removed from its environment.

use std::io::Read;
use std::net::TcpListener;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use oxibonsai_testkit::dense_fixture::{byte_tokenizer_json, weighted_dense_gguf};
use oxibonsai_testkit::mmproj_fixture::{synthetic_mmproj_gguf, MmprojFixtureSpec};
use oxibonsai_testkit::temp_path::TempFile;

/// A loopback listener that counts (and immediately drops) connections.
struct Counter {
    port: u16,
    accepts: Arc<AtomicUsize>,
}

impl Counter {
    fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind a loopback listener");
        let port = listener.local_addr().expect("address").port();
        let accepts = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&accepts);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                counted.fetch_add(1, Ordering::SeqCst);
                if let Ok(mut stream) = stream {
                    let mut buffer = [0u8; 512];
                    let _ = stream.set_read_timeout(Some(Duration::from_millis(200)));
                    let _ = stream.read(&mut buffer);
                }
            }
        });
        Self { port, accepts }
    }

    fn url(&self) -> String {
        format!("http://127.0.0.1:{}/img.png", self.port)
    }

    fn accepts(&self) -> usize {
        // Give a connection that was (wrongly) made a moment to be counted.
        std::thread::sleep(Duration::from_millis(100));
        self.accepts.load(Ordering::SeqCst)
    }
}

/// The test kit's tiny projector, written to a temporary file.
fn projector() -> TempFile {
    let bytes = synthetic_mmproj_gguf(&MmprojFixtureSpec::tiny()).expect("the projector builds");
    TempFile::write("image_url_fetch_cli_mmproj", ".gguf", &bytes).expect("write the projector")
}

/// A model path that does not exist: the command stops when it opens it.
fn missing_model() -> String {
    std::env::temp_dir()
        .join("oxibonsai-image-url-fetch-no-such-model.gguf")
        .to_string_lossy()
        .into_owned()
}

/// `oxibonsai <args>` in a scratch directory, without the remote-image
/// variables (or any proxy) in its environment, plus `env`.
fn oxibonsai(args: &[&str], env: &[(&str, &str)]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_oxibonsai"));
    command
        .args(args)
        .current_dir(std::env::temp_dir())
        .env_remove("OXI_ALLOW_IMAGE_URL_FETCH")
        .env_remove("OXI_IMAGE_URL_TIMEOUT_MS")
        .env_remove("OXI_IMAGE_URL_ALLOW_HOSTS")
        .env_remove("OXI_MEDIA_PATH")
        .env_remove("OXI_MODEL");
    for (key, value) in env {
        command.env(key, value);
    }
    command.output().expect("run the oxibonsai binary")
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn run_with_image(image: &str, extra: &[&str], env: &[(&str, &str)]) -> Output {
    let mmproj = projector();
    let model = missing_model();
    let mmproj_path = mmproj.path().to_string_lossy().into_owned();
    let mut args = vec![
        "run",
        "--model",
        model.as_str(),
        "--mmproj",
        mmproj_path.as_str(),
        "--image",
        image,
        "-p",
        "Describe this image.",
    ];
    args.extend_from_slice(extra);
    oxibonsai(&args, env)
}

// ── Without the opt-in ───────────────────────────────────────────────────

#[test]
fn without_the_opt_in_a_remote_image_is_refused_and_nothing_is_opened() {
    let listener = Counter::start();
    let output = run_with_image(&listener.url(), &[], &[]);
    let text = stderr(&output);
    assert!(!output.status.success(), "{text}");
    assert!(text.contains("[image_url_fetch_disabled]"), "{text}");
    assert!(text.contains("--allow-image-url-fetch"), "{text}");
    assert_eq!(listener.accepts(), 0);
}

// ── With the opt-in: validated, never fetched before the model loads ─────

#[test]
fn with_the_opt_in_a_loopback_image_is_refused_by_the_address_policy() {
    let listener = Counter::start();
    for opt_in in [
        (vec!["--allow-image-url-fetch"], vec![]),
        (vec![], vec![("OXI_ALLOW_IMAGE_URL_FETCH", "1")]),
    ] {
        let output = run_with_image(&listener.url(), &opt_in.0, &opt_in.1);
        let text = stderr(&output);
        assert!(!output.status.success(), "{text}");
        assert!(text.contains("[image_url_refused]"), "{text}");
        assert!(text.contains("not a public address"), "{text}");
    }
    assert_eq!(listener.accepts(), 0);
}

#[test]
fn an_allowlisted_image_passes_validation_without_being_fetched() {
    let listener = Counter::start();
    let allow = format!("127.0.0.1:{}", listener.port);
    let output = run_with_image(
        &listener.url(),
        &[
            "--allow-image-url-fetch",
            "--image-url-allow-host",
            allow.as_str(),
            "--image-url-timeout-ms",
            "2000",
        ],
        &[],
    );
    let text = stderr(&output);
    assert!(!output.status.success(), "the model does not exist: {text}");
    assert!(
        !text.contains("[image_url"),
        "the image passed validation: {text}"
    );
    assert_eq!(listener.accepts(), 0, "validation never fetches");
}

// ── Settings refused at start-up, by name ────────────────────────────────

/// Flags, environment, and the words the refusal must contain.
type Case = (
    Vec<&'static str>,
    Vec<(&'static str, &'static str)>,
    &'static str,
);

#[test]
fn malformed_settings_are_refused_at_startup_naming_the_setting() {
    let listener = Counter::start();
    let url = listener.url();
    let cases: Vec<Case> = vec![
        (
            vec![
                "--allow-image-url-fetch",
                "--image-url-allow-host",
                "bad host",
            ],
            vec![],
            "--image-url-allow-host",
        ),
        (
            vec![
                "--allow-image-url-fetch",
                "--image-url-allow-host",
                "images.intranet:0",
            ],
            vec![],
            "--image-url-allow-host",
        ),
        (
            vec!["--allow-image-url-fetch", "--image-url-timeout-ms", "0"],
            vec![],
            "--image-url-timeout-ms",
        ),
        (
            vec!["--allow-image-url-fetch", "--image-url-timeout-ms", "soon"],
            vec![],
            "--image-url-timeout-ms",
        ),
        (
            vec!["--allow-image-url-fetch"],
            vec![("OXI_IMAGE_URL_TIMEOUT_MS", "ten")],
            "OXI_IMAGE_URL_TIMEOUT_MS",
        ),
        (
            vec!["--allow-image-url-fetch"],
            vec![("OXI_IMAGE_URL_ALLOW_HOSTS", "ok.example,user@bad")],
            "OXI_IMAGE_URL_ALLOW_HOSTS",
        ),
        (
            vec!["--image-url-allow-host", "images.intranet"],
            vec![],
            "--image-url-allow-host has no effect without --allow-image-url-fetch",
        ),
        (
            vec![],
            vec![("OXI_IMAGE_URL_ALLOW_HOSTS", "images.intranet")],
            "OXI_IMAGE_URL_ALLOW_HOSTS is set, but remote image fetching is not enabled",
        ),
    ];
    for (flags, env, names) in cases {
        let output = run_with_image(&url, &flags, &env);
        let text = stderr(&output);
        assert!(!output.status.success(), "{flags:?} {env:?}: {text}");
        assert!(
            text.contains(names),
            "{flags:?} {env:?}: expected {names:?} in: {text}"
        );
    }
    assert_eq!(listener.accepts(), 0);
}

// ── A text-only command never reads the remote-image settings ────────────

/// Remote-image environments a command that loads a projector refuses at
/// start-up, each with the words of its refusal: an allowlist without the
/// opt-in, and the opt-in with a deadline that is not a number.
const STALE_SETTINGS: [(&[(&str, &str)], &str); 2] = [
    (
        &[("OXI_IMAGE_URL_ALLOW_HOSTS", "images.intranet:8080")],
        "OXI_IMAGE_URL_ALLOW_HOSTS is set, but remote image fetching is not enabled",
    ),
    (
        &[
            ("OXI_ALLOW_IMAGE_URL_FETCH", "1"),
            ("OXI_IMAGE_URL_TIMEOUT_MS", "abc"),
        ],
        "OXI_IMAGE_URL_TIMEOUT_MS=\"abc\" is not a number of milliseconds",
    ),
];

/// The test kit's small dense (`qwen3`) model and its byte-level tokenizer,
/// written to temporary files: a model `run` and `chat` load in a moment.
struct DenseModel {
    model: TempFile,
    tokenizer: TempFile,
}

impl DenseModel {
    fn write() -> Self {
        Self {
            model: TempFile::write("image_url_fetch_cli_dense", ".gguf", &weighted_dense_gguf())
                .expect("write the dense model"),
            tokenizer: TempFile::write(
                "image_url_fetch_cli_tokenizer",
                ".json",
                byte_tokenizer_json().as_bytes(),
            )
            .expect("write the tokenizer"),
        }
    }

    /// `<command> --model <model> --tokenizer <tokenizer> --backend cpu
    /// --max-tokens 2`, plus a one-word prompt for `run` (`chat` reads its
    /// turns from stdin, which `Command::output` closes, so the session ends
    /// at once) and `extra`.
    fn args(&self, command: &str, extra: &[&str]) -> Vec<String> {
        let mut args = vec![
            command.to_string(),
            "--model".to_string(),
            self.model.path().to_string_lossy().into_owned(),
            "--tokenizer".to_string(),
            self.tokenizer.path().to_string_lossy().into_owned(),
            "--backend".to_string(),
            "cpu".to_string(),
            "--max-tokens".to_string(),
            "2".to_string(),
        ];
        if command == "run" {
            args.extend(["-p".to_string(), "hi".to_string()]);
        }
        args.extend(extra.iter().map(ToString::to_string));
        args
    }
}

/// `oxibonsai <args>` as [`oxibonsai`] runs it, from owned arguments.
fn oxibonsai_owned(args: &[String], env: &[(&str, &str)]) -> Output {
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    oxibonsai(&args, env)
}

#[test]
fn a_text_only_command_ignores_stale_remote_image_settings() {
    let dense = DenseModel::write();
    let clean: &[(&str, &str)] = &[];
    let environments = std::iter::once(clean).chain(STALE_SETTINGS.iter().map(|&(env, _)| env));
    for env in environments {
        for command in ["run", "chat"] {
            let output = oxibonsai_owned(&dense.args(command, &[]), env);
            let text = stderr(&output);
            for (_, refusal) in STALE_SETTINGS {
                assert!(
                    !text.contains(refusal),
                    "`{command}` without --mmproj read the remote-image settings {env:?}: {text}"
                );
            }
            assert!(
                output.status.success(),
                "`{command}` without --mmproj must not be stopped by {env:?}: {text}"
            );
            assert!(
                text.contains("Resolved model:"),
                "`{command}` without --mmproj must load the model under {env:?}: {text}"
            );
        }
    }
}

#[test]
fn with_a_projector_stale_remote_image_settings_stop_the_command_before_the_model_loads() {
    let dense = DenseModel::write();
    let mmproj = projector();
    let mmproj_path = mmproj.path().to_string_lossy().into_owned();
    for (env, refusal) in STALE_SETTINGS {
        for command in ["run", "chat"] {
            let output = oxibonsai_owned(
                &dense.args(command, &["--mmproj", mmproj_path.as_str()]),
                env,
            );
            let text = stderr(&output);
            assert!(
                !output.status.success(),
                "`{command} --mmproj` under {env:?}: {text}"
            );
            assert!(
                text.contains(refusal),
                "`{command} --mmproj` under {env:?}: expected {refusal:?} in: {text}"
            );
            assert!(
                !text.contains("Resolved model:"),
                "`{command} --mmproj` under {env:?} is refused before the model loads: {text}"
            );
        }
    }
}

// ── What --help says ─────────────────────────────────────────────────────

#[test]
fn the_help_says_what_is_fetched_and_what_is_refused() {
    let mut commands = vec!["run", "chat"];
    if cfg!(feature = "server") {
        commands.push("serve");
    }
    for command in commands {
        let output = oxibonsai(&[command, "--help"], &[]);
        assert!(output.status.success(), "{command} --help");
        let help = String::from_utf8_lossy(&output.stdout).into_owned();
        for expected in [
            "--allow-image-url-fetch",
            "--image-url-timeout-ms",
            "--image-url-allow-host",
            "OXI_ALLOW_IMAGE_URL_FETCH",
            "OXI_IMAGE_URL_TIMEOUT_MS",
            "OXI_IMAGE_URL_ALLOW_HOSTS",
            "image_url_fetch_disabled",
            "public",
        ] {
            assert!(
                help.contains(expected),
                "{command} --help lacks {expected:?}:\n{help}"
            );
        }
        for stale in [
            "never fetched",
            "still refused",
            "only changes the reason",
            "has no fetcher",
        ] {
            assert!(
                !help.contains(stale),
                "{command} --help still says {stale:?}:\n{help}"
            );
        }
    }
}
