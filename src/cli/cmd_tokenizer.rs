//! `oxibonsai tokenizer` — manage the Qwen3 tokenizer (download / inspect).

use super::args::TokenizerCmd;
use std::time::Duration;

/// HTTP timeout applied to tokenizer downloads (cli-M3). `tokenizer.json`
/// files (vocab + merges) are at most a few tens of MB; a well-behaved server
/// on any working connection finishes comfortably inside this window, so an
/// unresponsive or hung server is treated as a failure instead of blocking
/// the CLI indefinitely.
const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(30);

pub(crate) fn run(tok_cmd: TokenizerCmd) -> anyhow::Result<()> {
    match tok_cmd {
        TokenizerCmd::Download {
            output,
            repo,
            force,
        } => {
            let out_path = std::path::Path::new(&output);
            if out_path.exists() && !force {
                println!("tokenizer.json already exists at {output}");
                println!("Use --force to overwrite.");
                return Ok(());
            }
            if let Some(parent) = out_path.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent).map_err(|e| {
                        anyhow::anyhow!("failed to create directory {}: {e}", parent.display())
                    })?;
                }
            }
            let url = format!("https://huggingface.co/{repo}/resolve/main/tokenizer.json");
            println!("Downloading tokenizer.json from {url}");
            let bytes = fetch_url_bytes(&url, DOWNLOAD_TIMEOUT)?;
            // cli-M3: validate before writing, so a redirected-to-an-error-page
            // or truncated download never silently clobbers a working file.
            validate_tokenizer_json(&bytes).map_err(|e| {
                anyhow::anyhow!(
                    "refusing to write {output}: downloaded content does not look like a \
                     tokenizer.json ({e})"
                )
            })?;
            std::fs::write(out_path, &bytes)
                .map_err(|e| anyhow::anyhow!("failed to write {output}: {e}"))?;
            println!("Saved to {output} ({} KB)", bytes.len() / 1024);
        }

        TokenizerCmd::Info { path } => {
            let data = std::fs::read_to_string(&path)
                .map_err(|e| anyhow::anyhow!("cannot read {path}: {e}"))?;
            let v: serde_json::Value = serde_json::from_str(&data)
                .map_err(|e| anyhow::anyhow!("invalid JSON in {path}: {e}"))?;
            let model_type = v
                .get("model")
                .and_then(|m| m.get("type"))
                .and_then(|t| t.as_str())
                .unwrap_or("unknown");
            let vocab_size = v
                .get("model")
                .and_then(|m| m.get("vocab"))
                .map(|vocab| {
                    if let Some(obj) = vocab.as_object() {
                        obj.len()
                    } else {
                        0
                    }
                })
                .unwrap_or(0);
            let added_tokens = v
                .get("added_tokens")
                .and_then(|t| t.as_array())
                .map(|a| a.len())
                .unwrap_or(0);
            println!("Tokenizer: {path}");
            println!("  Type:         {model_type}");
            println!("  Vocab size:   {vocab_size}");
            println!("  Added tokens: {added_tokens}");
        }
    }

    Ok(())
}

/// Download `url`'s body over Pure-Rust HTTP/HTTPS (`oxihttp`; deps-01),
/// failing after `timeout` instead of hanging indefinitely (cli-M3).
///
/// `run` above is a synchronous entry point but `oxihttp`'s client API is
/// async-only, so this spins up a short-lived, single-threaded runtime for
/// the one request and tears it down again.
///
/// That runtime is built, driven, and dropped entirely on a dedicated OS
/// thread (via `std::thread::scope`), never on the caller's own thread.
/// `fetch_url_bytes` is reached two different ways: directly from unit tests
/// (no ambient tokio runtime), and from the real `oxibonsai tokenizer
/// download` binary path, where `main.rs` builds a multi-thread tokio
/// runtime and `block_on`s `cli::run()`, which calls this function
/// synchronously from *inside* that runtime. Two separate tokio panics are
/// possible when a runtime's construction/use/drop happens on a thread that
/// is already inside another runtime's context:
///   - `Runtime::block_on` panics with "Cannot start a runtime from within a
///     runtime" if called on such a thread.
///   - `Runtime::drop` panics with "Cannot drop a runtime in a context where
///     blocking is not allowed" if the ambient runtime is a `current_thread`
///     one (this does not fire under `main.rs`'s `multi_thread` runtime,
///     which allows block-in-place drops — but it would under, say, a plain
///     `#[tokio::test]`, which defaults to `current_thread`).
///
/// Building, using, and dropping `rt` entirely inside the spawned thread
/// sidesteps both, regardless of what (if any) runtime the caller is on.
fn fetch_url_bytes(url: &str, timeout: Duration) -> anyhow::Result<Vec<u8>> {
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|e| anyhow::anyhow!("failed to start async runtime: {e}"))?;
                rt.block_on(fetch_url_bytes_async(url, timeout))
            })
            .join()
            .map_err(|_| anyhow::anyhow!("download thread panicked"))?
    })
}

async fn fetch_url_bytes_async(url: &str, timeout: Duration) -> anyhow::Result<Vec<u8>> {
    // `build_https()` (not `build()`) is required here: a plain `Client` never
    // negotiates TLS, so it cannot complete an `https://` request at all.
    // `.with_tls()` is *also* required (not just "https": true by default):
    // without it `build_https()` fails immediately with "at least one root
    // store ... must be configured" — verified empirically — because
    // `oxitls`'s TLS 1.3 client builder has no trust store configured until
    // something enables one. `.with_tls()` turns on the Mozilla CA bundle
    // (webpki-roots), which is what a normal HTTPS client needs to verify
    // huggingface.co's certificate.
    let client = oxihttp::Client::builder()
        .with_tls()
        .build_https()
        .map_err(|e| anyhow::anyhow!("failed to build HTTP client: {e}"))?;
    let response = client
        .get(url)
        .map_err(|e| anyhow::anyhow!("invalid URL {url}: {e}"))?
        .timeout(timeout)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("HTTP request failed: {e}"))?;
    if !response.status().is_success() {
        anyhow::bail!("server returned {} for {url}", response.status());
    }
    let bytes = response
        .body_bytes()
        .await
        .map_err(|e| anyhow::anyhow!("failed to read response body: {e}"))?;
    Ok(bytes.to_vec())
}

/// Reject anything that does not have the minimal shape of a HF
/// `tokenizer.json` (cli-M3): a JSON object with a `model` object. Runs
/// before the bytes are written to disk.
fn validate_tokenizer_json(bytes: &[u8]) -> anyhow::Result<()> {
    let value: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|e| anyhow::anyhow!("not valid JSON: {e}"))?;
    let model = value
        .get("model")
        .ok_or_else(|| anyhow::anyhow!("missing top-level \"model\" field"))?;
    if !model.is_object() {
        anyhow::bail!("\"model\" field is not a JSON object");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    #[test]
    fn validate_tokenizer_json_accepts_minimal_shape() {
        let body = br#"{"model": {"type": "BPE", "vocab": {}}}"#;
        validate_tokenizer_json(body).unwrap();
    }

    #[test]
    fn validate_tokenizer_json_rejects_non_json() {
        let err = validate_tokenizer_json(b"not json at all").unwrap_err();
        assert!(err.to_string().contains("not valid JSON"));
    }

    #[test]
    fn validate_tokenizer_json_rejects_html_error_page() {
        // The class of bug this guards against: a redirected/expired URL
        // serving an HTML error page instead of a tokenizer.json.
        let err = validate_tokenizer_json(b"<html><body>404 Not Found</body></html>").unwrap_err();
        assert!(err.to_string().contains("not valid JSON"));
    }

    #[test]
    fn validate_tokenizer_json_rejects_json_without_model() {
        let err = validate_tokenizer_json(br#"{"hello": "world"}"#).unwrap_err();
        assert!(err.to_string().contains("model"));
    }

    #[test]
    fn validate_tokenizer_json_rejects_non_object_model() {
        let err = validate_tokenizer_json(br#"{"model": "not an object"}"#).unwrap_err();
        assert!(err.to_string().contains("model"));
    }

    /// cli-M3: a server that accepts the connection but never answers must
    /// not hang the CLI forever — the request has to fail once `timeout`
    /// elapses. Local-only TCP listener; no real network access.
    #[test]
    fn fetch_url_bytes_times_out_on_unresponsive_server() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind local listener");
        let addr = listener.local_addr().expect("read local addr");
        let _server = std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                // Read whatever the client sent, then go quiet: the response
                // never comes, so the client-side timeout must fire. The
                // sleep just keeps the socket open a little longer than the
                // timeout under test; the thread is not joined, and the
                // whole test process exits (and reaps it) long before it
                // would return on its own.
                let mut buf = [0u8; 1024];
                let _ = stream.read(&mut buf);
                std::thread::sleep(Duration::from_secs(2));
            }
        });

        let url = format!("http://{addr}/tokenizer.json");
        let result = fetch_url_bytes(&url, Duration::from_millis(200));

        let err = result.expect_err("expected a timeout error, got a successful download");
        assert!(
            err.to_string().to_lowercase().contains("timeout"),
            "expected a timeout-flavored error, got: {err}"
        );
    }

    #[test]
    fn fetch_url_bytes_returns_body_on_success() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind local listener");
        let addr = listener.local_addr().expect("read local addr");
        let body: &[u8] = br#"{"model": {"type": "BPE", "vocab": {}}}"#;
        let server = std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf);
                let mut response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                )
                .into_bytes();
                response.extend_from_slice(body);
                let _ = stream.write_all(&response);
                let _ = stream.flush();
            }
        });

        let url = format!("http://{addr}/tokenizer.json");
        let result = fetch_url_bytes(&url, Duration::from_secs(5));
        // Check the client result before joining: if the client failed
        // without ever connecting, the server thread would sit at
        // `accept()` forever and `join()` would hang, turning a clear
        // assertion failure into an unexplained test timeout.
        let got = result.expect("expected a successful download");
        server.join().expect("fake server thread panicked");
        assert_eq!(got, body);
    }

    /// Regression test for "Cannot start a runtime from within a runtime":
    /// `fetch_url_bytes` must work when called from code that is already
    /// being driven by a tokio runtime, exactly like the real `oxibonsai
    /// tokenizer download` binary path (`main.rs` builds a multi-thread
    /// runtime and `block_on`s `cli::run()`, which calls
    /// `cmd_tokenizer::run` -> `fetch_url_bytes` synchronously from inside
    /// it). The other tests above call `fetch_url_bytes` with no ambient
    /// runtime at all, so they cannot catch this class of bug.
    #[tokio::test(flavor = "multi_thread")]
    async fn fetch_url_bytes_works_inside_an_ambient_multi_thread_runtime() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind local listener");
        let addr = listener.local_addr().expect("read local addr");
        let body: &[u8] = br#"{"model": {"type": "BPE", "vocab": {}}}"#;
        let server = std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf);
                let mut response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                )
                .into_bytes();
                response.extend_from_slice(body);
                let _ = stream.write_all(&response);
                let _ = stream.flush();
            }
        });

        let url = format!("http://{addr}/tokenizer.json");
        // Before the `std::thread::scope` fix, this call panics with "Cannot
        // start a runtime from within a runtime" because this test function
        // is itself already executing on a multi-thread tokio runtime.
        let result = fetch_url_bytes(&url, Duration::from_secs(5));
        let got = result.expect("expected a successful download from inside an ambient runtime");
        server.join().expect("fake server thread panicked");
        assert_eq!(got, body);
    }
}
