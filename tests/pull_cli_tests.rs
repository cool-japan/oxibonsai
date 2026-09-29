//! Integration tests for `oxibonsai pull` (cli-05 / sec-12).
//!
//! Every test spawns the real compiled binary (`CARGO_BIN_EXE_oxibonsai`)
//! against a local, in-process `TcpListener` HTTP server — never the real
//! network. `OXIBONSAI_HF_BASE_URL` redirects a named entry's HuggingFace URL
//! to that server; `OXIBONSAI_CHECKSUMS_FILE` points at a per-test checksums
//! file. Every variable is set on the spawned CHILD only (`Command::env` /
//! `env_remove`), never on this test process, so the tests run safely in
//! parallel.
//!
//! A named entry is verified FAIL-CLOSED against the size and SHA-256
//! compiled into the binary, so a tiny fixture can never be accepted as a
//! real 27B file: the named-entry tests prove each refusal (wrong
//! architecture, wrong Hadamard contract, wrong size) end to end, from a
//! temporary working directory with no checksums file anywhere (the check
//! does not depend on the CWD). Resume and progress are exercised through
//! bare-URL downloads verified against a checksums file. The named-entry
//! happy path serves the REAL artifacts from `models/` through the local
//! server (`OXIBONSAI_MODELS_DIR` / `OXI_BONSAI2_PTQ1_GGUF`; skipped with a
//! capability report when unset), so the digests and sizes compiled into
//! the binary are proven against the real files.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::Command;

use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue, TensorEntry, TensorType};

// ── Shared helpers ──────────────────────────────────────────────────────────

fn scratch_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "oxibonsai_pull_cli_test_{tag}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
    ));
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

/// A small, well-formed GGUF with `general.architecture = arch` and, when
/// given, `prism.hadamard.version = v` (+ a minimal contract).
fn gguf_fixture(arch: &str, hadamard_version: Option<u32>) -> Vec<u8> {
    let mut writer = GgufWriter::new();
    writer.add_metadata(
        "general.architecture",
        MetadataWriteValue::Str(arch.to_string()),
    );
    if let Some(version) = hadamard_version {
        writer.add_metadata("prism.hadamard.version", MetadataWriteValue::U32(version));
        writer.add_metadata("prism.hadamard.block_size", MetadataWriteValue::U32(8));
        writer.add_metadata(
            "prism.hadamard.weight_names",
            MetadataWriteValue::ArrayStr(vec!["token_embd.weight".to_string()]),
        );
    }
    let hidden = 8u64;
    let vocab = 64u64;
    let embd_data: Vec<u8> = (0..(hidden * vocab))
        .flat_map(|i| ((i as f32) * 0.01).to_le_bytes())
        .collect();
    writer.add_tensor(TensorEntry {
        name: "token_embd.weight".to_string(),
        shape: vec![hidden, vocab],
        tensor_type: TensorType::F32,
        data: embd_data,
    });
    writer.to_bytes().expect("build GGUF fixture")
}

/// What the test server answers for request `index` (0-based), given the
/// `Range: bytes=N-` start the client sent (if any): the raw HTTP response
/// bytes, and whether to cut the connection after `cut_after` body bytes.
struct Reply {
    raw: Vec<u8>,
}

fn ok_full(body: &[u8]) -> Reply {
    let mut raw = format!(
        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )
    .into_bytes();
    raw.extend_from_slice(body);
    Reply { raw }
}

fn partial(body: &[u8], start: usize, advertised_start: usize) -> Reply {
    let slice = &body[start..];
    let mut raw = format!(
        "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes \
         {advertised_start}-{}/{}\r\nConnection: close\r\n\r\n",
        slice.len(),
        body.len() - 1,
        body.len()
    )
    .into_bytes();
    raw.extend_from_slice(slice);
    Reply { raw }
}

/// A 200 that advertises the full length but only sends `sent` body bytes
/// before closing (a connection dropped mid-transfer).
fn truncated_full(body: &[u8], sent: usize) -> Reply {
    let mut raw = format!(
        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )
    .into_bytes();
    raw.extend_from_slice(&body[..sent]);
    Reply { raw }
}

/// A local server answering up to `max_requests` connections with
/// `handler(index, range_start)`. Returns its address and a log of the range
/// start each request carried.
fn spawn_server(
    max_requests: usize,
    handler: impl Fn(usize, Option<usize>) -> Reply + Send + 'static,
) -> (
    std::net::SocketAddr,
    std::sync::Arc<std::sync::Mutex<Vec<Option<usize>>>>,
) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind local listener");
    let addr = listener.local_addr().expect("read local addr");
    let log = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let log_thread = std::sync::Arc::clone(&log);
    std::thread::spawn(move || {
        for index in 0..max_requests {
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
            let mut buf = [0u8; 8192];
            let n = stream.read(&mut buf).unwrap_or(0);
            let request = String::from_utf8_lossy(&buf[..n]);
            let range_start = request
                .lines()
                .find(|l| l.to_ascii_lowercase().starts_with("range:"))
                .and_then(|l| l.split("bytes=").nth(1))
                .and_then(|s| s.trim_end_matches('-').trim().parse::<usize>().ok());
            if let Ok(mut log) = log_thread.lock() {
                log.push(range_start);
            }
            let reply = handler(index, range_start);
            let _ = stream.write_all(&reply.raw);
            let _ = stream.flush();
        }
    });
    (addr, log)
}

fn write_checksums_file(dir: &std::path::Path, filename: &str, hex: &str) -> PathBuf {
    let path = dir.join("checksums.sha256");
    std::fs::write(&path, format!("{hex}  models/{filename}\n")).expect("write checksums file");
    path
}

fn sha256_hex(bytes: &[u8]) -> String {
    let tmp = std::env::temp_dir().join(format!(
        "oxibonsai_pull_test_hash_scratch_{}_{}.bin",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::write(&tmp, bytes).expect("write scratch file for hashing");
    let hex = oxibonsai_runtime::serve_shared::compute_sha256_hex(&tmp)
        .expect("hash scratch file")
        .expect("digest always present for a readable file");
    let _ = std::fs::remove_file(&tmp);
    hex
}

/// The binary with every ambient pull-related variable cleared.
fn pull_command(dir: &std::path::Path) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_oxibonsai"));
    cmd.current_dir(dir)
        .env_remove("OXIBONSAI_HF_BASE_URL")
        .env_remove("OXIBONSAI_CHECKSUMS_FILE")
        .env_remove("OXI_BONSAI2_REPO")
        .env_remove("OXI_BONSAI2_DEV_REPO");
    cmd
}

// ── Tests ────────────────────────────────────────────────────────────────

#[test]
fn pull_help_advertises_the_subcommand_and_its_flags() {
    let output = Command::new(env!("CARGO_BIN_EXE_oxibonsai"))
        .args(["pull", "--help"])
        .output()
        .expect("failed to spawn oxibonsai binary");
    assert!(
        output.status.success(),
        "`pull --help` should succeed; stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    for flag in ["--out", "--force", "--band", "--vision"] {
        assert!(
            stdout.contains(flag),
            "`pull --help` must advertise {flag}; got:\n{stdout}"
        );
    }
    assert!(
        stdout.contains("FAIL-CLOSED"),
        "help must state the verification contract"
    );
}

#[test]
fn pull_rejects_an_unknown_target_naming_valid_choices() {
    let dir = scratch_dir("unknown_target");
    let output = pull_command(&dir)
        .args(["pull", "not-a-real-model-name", "--out"])
        .arg(&dir)
        .output()
        .expect("failed to spawn oxibonsai binary");
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        !output.status.success(),
        "an unknown target must be a hard error"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("bonsai2-27b-pq2_0"),
        "error should list a valid named entry; got: {stderr}"
    );
}

/// Happy path over a bare URL: streamed, verified against the checksums
/// file, renamed into place, with a live progress line on stderr.
#[test]
fn pull_downloads_verifies_and_saves_a_url_download_with_progress() {
    let dir = scratch_dir("happy_path");
    let body = gguf_fixture("qwen3", None);
    let checksums = write_checksums_file(&dir, "Custom-Model.gguf", &sha256_hex(&body));
    let served = body.clone();
    let (addr, _) = spawn_server(1, move |_, _| ok_full(&served));

    let output = pull_command(&dir)
        .args([
            "pull",
            &format!("http://{addr}/owner/repo/Custom-Model.gguf"),
            "--out",
        ])
        .arg(&dir)
        .env("OXIBONSAI_CHECKSUMS_FILE", &checksums)
        .output()
        .expect("failed to spawn oxibonsai binary");

    let saved = std::fs::read(dir.join("Custom-Model.gguf"));
    let part_exists = dir.join("Custom-Model.gguf.part").exists();
    let _ = std::fs::remove_dir_all(&dir);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "pull should succeed; stderr={stderr}"
    );
    assert_eq!(saved.expect("saved file must exist"), body);
    assert!(
        !part_exists,
        "the .part file must be renamed away on success"
    );
    assert!(
        stderr.contains("100.0%"),
        "a progress line must be printed: {stderr}"
    );
    assert!(stderr.contains("checksum OK"), "{stderr}");
}

#[test]
fn pull_refuses_a_bad_magic_payload_and_leaves_no_accepted_file() {
    let dir = scratch_dir("bad_magic");
    let body = b"this is not a gguf file, just garbage bytes".to_vec();
    let (addr, _) = spawn_server(1, move |_, _| ok_full(&body));

    let output = pull_command(&dir)
        .args(["pull", "bonsai2-27b-pq2_0", "--out"])
        .arg(&dir)
        .env("OXIBONSAI_HF_BASE_URL", format!("http://{addr}"))
        .output()
        .expect("failed to spawn oxibonsai binary");

    let saved_exists = dir.join("Ternary-Bonsai-2-27B-PQ2_0.gguf").exists();
    let part_exists = dir.join("Ternary-Bonsai-2-27B-PQ2_0.gguf.part").exists();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        !output.status.success(),
        "a bad-magic payload must be refused"
    );
    assert!(!saved_exists, "no file may be accepted at the final path");
    assert!(!part_exists, "the rejected .part file must be cleaned up");
    let stderr = String::from_utf8_lossy(&output.stderr).to_lowercase();
    assert!(
        stderr.contains("gguf") || stderr.contains("magic"),
        "error should explain the structural rejection; got: {stderr}"
    );
}

/// A named entry is verified against the digest and size compiled into the
/// binary — with NO checksums file reachable (a temp CWD, the variable
/// unset): a well-formed but wrong file is refused on its size, fail-closed.
#[test]
fn pull_named_entry_fails_closed_without_any_checksums_file() {
    let dir = scratch_dir("fail_closed");
    let body = gguf_fixture("qwen35", Some(1));
    let (addr, _) = spawn_server(1, move |_, _| ok_full(&body));

    let output = pull_command(&dir)
        .args(["pull", "bonsai2-27b-pq2_0", "--out"])
        .arg(&dir)
        .env("OXIBONSAI_HF_BASE_URL", format!("http://{addr}"))
        .output()
        .expect("failed to spawn oxibonsai binary");

    let saved_exists = dir.join("Ternary-Bonsai-2-27B-PQ2_0.gguf").exists();
    let part_exists = dir.join("Ternary-Bonsai-2-27B-PQ2_0.gguf.part").exists();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        !output.status.success(),
        "a wrong file must never be accepted"
    );
    assert!(!saved_exists && !part_exists, "nothing may be left behind");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("size MISMATCH"), "{stderr}");
}

#[test]
fn pull_named_entry_refuses_the_wrong_architecture() {
    let dir = scratch_dir("wrong_arch");
    let body = gguf_fixture("llama", Some(1));
    let (addr, _) = spawn_server(1, move |_, _| ok_full(&body));

    let output = pull_command(&dir)
        .args(["pull", "bonsai2-27b-ptq1_0", "--out"])
        .arg(&dir)
        .env("OXIBONSAI_HF_BASE_URL", format!("http://{addr}"))
        .output()
        .expect("failed to spawn oxibonsai binary");
    let _ = std::fs::remove_dir_all(&dir);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("expected 'qwen35'"), "{stderr}");
}

#[test]
fn pull_named_entry_refuses_a_hadamard_version_other_than_1() {
    let dir = scratch_dir("hadamard_v2");
    let body = gguf_fixture("qwen35", Some(2));
    let (addr, _) = spawn_server(1, move |_, _| ok_full(&body));

    let output = pull_command(&dir)
        .args(["pull", "bonsai2-27b-q2_0", "--out"])
        .arg(&dir)
        .env("OXIBONSAI_HF_BASE_URL", format!("http://{addr}"))
        .output()
        .expect("failed to spawn oxibonsai binary");
    let _ = std::fs::remove_dir_all(&dir);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("prism.hadamard.version = 2"), "{stderr}");
}

#[test]
fn pull_refuses_a_checksum_mismatch() {
    let dir = scratch_dir("bad_checksum");
    let body = gguf_fixture("qwen3", None);
    let checksums = write_checksums_file(&dir, "Custom-Model.gguf", &"0".repeat(64));
    let (addr, _) = spawn_server(1, move |_, _| ok_full(&body));

    let output = pull_command(&dir)
        .args([
            "pull",
            &format!("http://{addr}/x/Custom-Model.gguf"),
            "--out",
        ])
        .arg(&dir)
        .env("OXIBONSAI_CHECKSUMS_FILE", &checksums)
        .output()
        .expect("failed to spawn oxibonsai binary");
    let saved_exists = dir.join("Custom-Model.gguf").exists();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        !output.status.success(),
        "a checksum mismatch must be refused"
    );
    assert!(!saved_exists, "no file may be accepted at the final path");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("checksum MISMATCH"), "{stderr}");
}

#[test]
fn pull_requires_force_to_overwrite_an_existing_file() {
    let dir = scratch_dir("no_force");
    let existing = dir.join("Ternary-Bonsai-2-27B-PQ2_0.gguf");
    std::fs::write(&existing, b"pre-existing content").expect("write pre-existing file");

    let output = pull_command(&dir)
        .args(["pull", "bonsai2-27b-pq2_0", "--out"])
        .arg(&dir)
        .output()
        .expect("failed to spawn oxibonsai binary");

    let unchanged = std::fs::read(&existing).expect("file still exists");
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        !output.status.success(),
        "must refuse to overwrite without --force"
    );
    assert_eq!(
        unchanged, b"pre-existing content",
        "existing file must be untouched"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--force"),
        "error should mention --force; got: {stderr}"
    );
}

/// A `.part` left behind by an interrupted run drives a `Range` request;
/// the `206` tail is appended (not the whole body re-fetched).
#[test]
fn pull_resumes_from_an_existing_part_file_via_range() {
    let dir = scratch_dir("resume");
    let body = gguf_fixture("qwen3", None);
    let split = body.len() / 2;
    let checksums = write_checksums_file(&dir, "Custom-Model.gguf", &sha256_hex(&body));
    std::fs::write(dir.join("Custom-Model.gguf.part"), &body[..split]).expect("pre-populate");

    let served = body.clone();
    let (addr, log) = spawn_server(1, move |_, range| match range {
        Some(start) => partial(&served, start, start),
        None => ok_full(&served),
    });
    let output = pull_command(&dir)
        .args([
            "pull",
            &format!("http://{addr}/x/Custom-Model.gguf"),
            "--out",
        ])
        .arg(&dir)
        .env("OXIBONSAI_CHECKSUMS_FILE", &checksums)
        .output()
        .expect("failed to spawn oxibonsai binary");

    let saved = std::fs::read(dir.join("Custom-Model.gguf"));
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        output.status.success(),
        "the resumed download should complete; stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        saved.expect("saved"),
        body,
        "the appended tail must reproduce the file"
    );
    assert_eq!(log.lock().expect("log").clone(), vec![Some(split)]);
}

/// A `206` whose `Content-Range` does NOT start at the resume offset is
/// never appended: the client restarts from byte 0 (no Range) and the result
/// is still exact.
#[test]
fn pull_restarts_when_the_206_content_range_does_not_match_the_resume_offset() {
    let dir = scratch_dir("bad_range");
    let body = gguf_fixture("qwen3", None);
    let split = body.len() / 2;
    let checksums = write_checksums_file(&dir, "Custom-Model.gguf", &sha256_hex(&body));
    std::fs::write(dir.join("Custom-Model.gguf.part"), &body[..split]).expect("pre-populate");

    let served = body.clone();
    let (addr, log) = spawn_server(2, move |_, range| match range {
        // Claims to start at 0 although the client asked for `start`.
        Some(start) => partial(&served, start, 0),
        None => ok_full(&served),
    });
    let output = pull_command(&dir)
        .args([
            "pull",
            &format!("http://{addr}/x/Custom-Model.gguf"),
            "--out",
        ])
        .arg(&dir)
        .env("OXIBONSAI_CHECKSUMS_FILE", &checksums)
        .output()
        .expect("failed to spawn oxibonsai binary");

    let saved = std::fs::read(dir.join("Custom-Model.gguf"));
    let _ = std::fs::remove_dir_all(&dir);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "stderr={stderr}");
    assert_eq!(saved.expect("saved"), body);
    assert!(stderr.contains("restarting from the beginning"), "{stderr}");
    assert_eq!(log.lock().expect("log").clone(), vec![Some(split), None]);
}

/// A connection that drops mid-body keeps what was streamed to disk and
/// resumes with a `Range` request for exactly the rest.
#[test]
fn pull_resumes_after_a_connection_dropped_mid_transfer() {
    let dir = scratch_dir("mid_drop");
    let body = gguf_fixture("qwen3", None);
    let cut = body.len() / 3;
    let checksums = write_checksums_file(&dir, "Custom-Model.gguf", &sha256_hex(&body));

    let served = body.clone();
    let (addr, log) = spawn_server(2, move |index, range| match (index, range) {
        (0, _) => truncated_full(&served, cut),
        (_, Some(start)) => partial(&served, start, start),
        (_, None) => ok_full(&served),
    });
    let output = pull_command(&dir)
        .args([
            "pull",
            &format!("http://{addr}/x/Custom-Model.gguf"),
            "--out",
        ])
        .arg(&dir)
        .env("OXIBONSAI_CHECKSUMS_FILE", &checksums)
        .output()
        .expect("failed to spawn oxibonsai binary");

    let saved = std::fs::read(dir.join("Custom-Model.gguf"));
    let _ = std::fs::remove_dir_all(&dir);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "stderr={stderr}");
    assert_eq!(saved.expect("saved"), body);
    assert!(stderr.contains("resuming"), "{stderr}");
    assert_eq!(log.lock().expect("log").clone(), vec![None, Some(cut)]);
}

// ── Named-entry happy path on the REAL artifacts (env-gated) ────────────────

/// A local server answering ONE request with the whole of `path`, streamed
/// from disk (the real artifacts are hundreds of MiB to several GiB).
fn spawn_file_server(path: PathBuf) -> std::net::SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind local listener");
    let addr = listener.local_addr().expect("read local addr");
    std::thread::spawn(move || {
        let Ok((mut stream, _)) = listener.accept() else {
            return;
        };
        let mut buf = [0u8; 8192];
        let _ = stream.read(&mut buf);
        let Ok(mut file) = std::fs::File::open(&path) else {
            let _ = stream.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n");
            return;
        };
        let len = file.metadata().map(|m| m.len()).unwrap_or(0);
        let head = format!("HTTP/1.1 200 OK\r\nContent-Length: {len}\r\nConnection: close\r\n\r\n");
        if stream.write_all(head.as_bytes()).is_ok() {
            let _ = std::io::copy(&mut file, &mut stream);
            let _ = stream.flush();
        }
    });
    addr
}

/// A real artifact located through an environment variable (a directory
/// plus a file name, or a direct file path), or `None` after a capability
/// report.
fn real_artifact(var: &str, file_in_dir: Option<&str>) -> Option<PathBuf> {
    let Some(value) = std::env::var_os(var).filter(|v| !v.is_empty()) else {
        eprintln!("SKIPPED (capability): set {var} to run the real-artifact pull check");
        return None;
    };
    let path = match file_in_dir {
        Some(name) => PathBuf::from(value).join(name),
        None => PathBuf::from(value),
    };
    if path.exists() {
        Some(path)
    } else {
        eprintln!("SKIPPED (capability): {} does not exist", path.display());
        None
    }
}

/// `oxibonsai pull <name>` of one real artifact through the local server,
/// from a temp CWD with no checksums file: returns the child's stderr after
/// asserting the file was verified and saved at its exact size.
fn pull_real_named_entry(name: &str, file_name: &str, source: PathBuf) -> String {
    let dir = scratch_dir(&format!("real_{name}"));
    let expected_len = std::fs::metadata(&source)
        .map(|m| m.len())
        .expect("stat source");
    let addr = spawn_file_server(source);
    let output = pull_command(&dir)
        .args(["pull", name, "--out"])
        .arg(&dir)
        .env("OXIBONSAI_HF_BASE_URL", format!("http://{addr}"))
        .output()
        .expect("failed to spawn oxibonsai binary");
    let saved = dir.join(file_name);
    let saved_len = std::fs::metadata(&saved).map(|m| m.len()).ok();
    let part_exists = dir.join(format!("{file_name}.part")).exists();
    let _ = std::fs::remove_dir_all(&dir);
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    // The verification transcript is the evidence this check exists for
    // (visible with `--nocapture`); `\r` progress refreshes become lines.
    eprintln!("--- pull {name} ---\n{}", stderr.replace('\r', "\n"));
    assert!(
        output.status.success(),
        "pull {name} should succeed; stderr={stderr}"
    );
    assert_eq!(
        saved_len,
        Some(expected_len),
        "{name} must be saved at its exact size"
    );
    assert!(
        !part_exists,
        "the .part file must be renamed away on success"
    );
    assert!(stderr.contains("Done:"), "{stderr}");
    stderr
}

/// The named-entry happy path end to end on the real artifacts: download,
/// structural check (architecture + Hadamard contract), exact size and the
/// SHA-256 compiled into the binary, then the rename into place.
///
/// * the Bonsai 2 mmproj (`OXIBONSAI_MODELS_DIR`) — the current upstream
///   digest (`checksum OK`), architecture `clip`;
/// * `Bonsai-8B.gguf` (`OXIBONSAI_MODELS_DIR`) — this checkout's copy is the
///   previously known-good object the legacy goldens were captured on, so it
///   is accepted as the named alternate WITH a warning;
/// * the Bonsai 2 27B PTQ1_0 (`OXI_BONSAI2_PTQ1_GGUF`) — `qwen35` with
///   `prism.hadamard.version = 1`, verified against the embedded digest.
#[test]
fn pull_downloads_verifies_and_saves_a_named_entry() {
    if let Some(source) = real_artifact(
        "OXIBONSAI_MODELS_DIR",
        Some("Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"),
    ) {
        let stderr = pull_real_named_entry(
            "bonsai2-27b-mmproj",
            "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf",
            source,
        );
        assert!(stderr.contains("general.architecture = 'clip'"), "{stderr}");
        assert!(stderr.contains("checksum OK (upstream"), "{stderr}");
    }
    if let Some(source) = real_artifact("OXIBONSAI_MODELS_DIR", Some("Bonsai-8B.gguf")) {
        let stderr = pull_real_named_entry("bonsai-8b", "Bonsai-8B.gguf", source);
        assert!(
            stderr.contains("checksum OK (upstream") || stderr.contains("previously known-good"),
            "Bonsai-8B must match the current upstream digest or the known-good alternate: \
             {stderr}"
        );
    }
    if let Some(source) = real_artifact("OXI_BONSAI2_PTQ1_GGUF", None) {
        let stderr = pull_real_named_entry(
            "bonsai2-27b-ptq1_0",
            "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
            source,
        );
        assert!(stderr.contains("prism.hadamard.version = 1 OK"), "{stderr}");
        assert!(stderr.contains("checksum OK (upstream"), "{stderr}");
    }
}
