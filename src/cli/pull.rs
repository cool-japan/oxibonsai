//! `oxibonsai pull` — download a Bonsai 2 / Bonsai model artifact from
//! HuggingFace, or an arbitrary URL, in Pure Rust (cli-05 / sec-12).
//!
//! `oxibonsai pull <name|url> [--out DIR] [--force]` over [`oxihttp`] (the
//! Pure-Rust, `oxitls`-backed client DEPS-PURE selected; the public webpki
//! roots are trusted for `https://`), with:
//!
//! * **streaming + resume** — the response body is streamed frame by frame
//!   straight into `<file>.part` (constant memory, never the whole
//!   multi-gigabyte body in RAM), so the final path's existence always means
//!   "a complete, verified download". A pre-existing `.part` drives an HTTP
//!   `Range` request; a `206` is accepted only when its `Content-Range`
//!   starts exactly at the resume offset (anything else restarts from byte
//!   0 rather than splicing the wrong bytes), and a plain `200` restarts from
//!   byte 0. A connection that drops mid-transfer keeps what was written and
//!   resumes, at most [`MAX_STALLED_ATTEMPTS`] times in a row without
//!   progress.
//! * **a live progress line** — bytes / total / percentage / throughput on
//!   stderr, refreshed as data arrives.
//! * **fail-closed verification (sec-12)** for a named entry, in this order:
//!   the GGUF structure (magic, `general.architecture` == the entry's
//!   expected architecture, and `prism.hadamard.version == 1` for a Bonsai 2
//!   language file), the exact byte size, then the SHA-256 — against digests
//!   COMPILED INTO this binary (from the upstream HuggingFace LFS objects),
//!   never a CWD-relative file, so an installed binary verifies exactly like
//!   a checkout. A mismatch refuses the file and deletes the `.part`.
//!   `scripts/checksums.sha256` / `OXIBONSAI_CHECKSUMS_FILE`, when readable,
//!   is an additional cross-checked authority, and the only hash authority
//!   for a bare URL.
//! * refuses to overwrite an existing file without `--force`.
//!
//! Named entries: the four Bonsai 2 27B artifacts (two repositories: the
//! main one and `-dev`, overridable via `OXI_BONSAI2_REPO` /
//! `OXI_BONSAI2_DEV_REPO` exactly like `scripts/download_ternary.sh`), the
//! three previous-generation 27B files, and `Bonsai-8B.gguf`.
//! `OXIBONSAI_HF_BASE_URL` points the whole manifest at a mirror (or a local
//! test server).
//!
//! Bonsai-8B: upstream re-uploaded the file
//! (same size) after this project's legacy goldens were captured. The
//! current upstream digest is the primary; the previously known-good digest
//! the goldens were captured on (`scripts/checksums.sha256`) is accepted as
//! a named alternate with a warning.

use std::io::Write;
use std::path::{Path, PathBuf};

/// `prism-ml/Ternary-Bonsai-2-27B-gguf` (confirmed by the project owner),
/// unless overridden by `OXI_BONSAI2_REPO`.
const HF_REPO_BONSAI2: &str = "prism-ml/Ternary-Bonsai-2-27B-gguf";
/// `prism-ml/Ternary-Bonsai-2-27B-gguf-dev` (the fork-only Q2_0 layout),
/// unless overridden by `OXI_BONSAI2_DEV_REPO`.
const HF_REPO_BONSAI2_DEV: &str = "prism-ml/Ternary-Bonsai-2-27B-gguf-dev";
/// The public HuggingFace endpoint, unless `OXIBONSAI_HF_BASE_URL` is set.
const HF_BASE_URL: &str = "https://huggingface.co";

/// Consecutive transfer attempts without a single new byte before `pull`
/// gives up.
pub(crate) const MAX_STALLED_ATTEMPTS: u32 = 5;

/// Which repository a manifest entry lives in — the two Bonsai 2 repos are
/// overridable via env; every other one is fixed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Repo {
    Bonsai2,
    Bonsai2Dev,
    Fixed(&'static str),
}

impl Repo {
    /// This repository given the two optional overrides (pure).
    fn resolve_with(self, bonsai2: Option<&str>, bonsai2_dev: Option<&str>) -> String {
        match self {
            Self::Bonsai2 => bonsai2.unwrap_or(HF_REPO_BONSAI2).to_string(),
            Self::Bonsai2Dev => bonsai2_dev.unwrap_or(HF_REPO_BONSAI2_DEV).to_string(),
            Self::Fixed(repo) => repo.to_string(),
        }
    }

    /// This repository with `OXI_BONSAI2_REPO` / `OXI_BONSAI2_DEV_REPO`
    /// applied.
    fn resolve(self) -> String {
        let primary = std::env::var("OXI_BONSAI2_REPO")
            .ok()
            .filter(|s| !s.is_empty());
        let dev = std::env::var("OXI_BONSAI2_DEV_REPO")
            .ok()
            .filter(|s| !s.is_empty());
        self.resolve_with(primary.as_deref(), dev.as_deref())
    }
}

/// One named, pre-registered download (design §5.8).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ManifestEntry {
    /// The name `oxibonsai pull <name>` matches, case-insensitively.
    pub(crate) name: &'static str,
    repo: Repo,
    /// File name within that repo.
    pub(crate) file: &'static str,
    /// Exact byte size (fail-closed).
    pub(crate) size: u64,
    /// SHA-256 of the current upstream object (fail-closed).
    pub(crate) sha256: &'static str,
    /// Previously known-good digests still accepted, with a warning.
    pub(crate) alternate_sha256: &'static [&'static str],
    /// The `general.architecture` the file must declare.
    pub(crate) expected_arch: &'static str,
    /// Whether the file must declare `prism.hadamard.version == 1` (every
    /// Hadamard-folded Bonsai 2 language GGUF).
    pub(crate) requires_hadamard: bool,
}

/// The download manifest. Sizes and digests are the upstream HuggingFace LFS
/// objects' own SHA-256, cross-checked against the repository's
/// `scripts/checksums.sha256` by
/// `tests::pull_manifest_agrees_with_the_repo_checksums_file`.
pub(crate) const MANIFEST: &[ManifestEntry] = &[
    ManifestEntry {
        name: "bonsai2-27b-ptq1_0",
        repo: Repo::Bonsai2,
        file: "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
        size: 5_946_648_928,
        sha256: "53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3",
        alternate_sha256: &[],
        expected_arch: "qwen35",
        requires_hadamard: true,
    },
    ManifestEntry {
        name: "bonsai2-27b-pq2_0",
        repo: Repo::Bonsai2,
        file: "Ternary-Bonsai-2-27B-PQ2_0.gguf",
        size: 7_206_168_928,
        sha256: "3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1",
        alternate_sha256: &[],
        expected_arch: "qwen35",
        requires_hadamard: true,
    },
    ManifestEntry {
        name: "bonsai2-27b-mmproj",
        repo: Repo::Bonsai2,
        file: "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf",
        size: 629_246_976,
        sha256: "6807ede61d570bb86ba34b756a0fa109edc33668604de867c6ea6d8f1d631903",
        alternate_sha256: &[],
        expected_arch: "clip",
        requires_hadamard: false,
    },
    // The fork-only group-128 Q2_0 layout (dev repo). Hadamard-folded like
    // the other Bonsai 2 language files (`prism.hadamard.version = 1`).
    ManifestEntry {
        name: "bonsai2-27b-q2_0",
        repo: Repo::Bonsai2Dev,
        file: "Ternary-Bonsai-2-27B-Q2_0-prism-fork-required.gguf",
        size: 7_626_008_928,
        sha256: "4f99aed01b8a877e153f9aa6569a4440fe17c59701cc0953e36d7f460549e70e",
        alternate_sha256: &[],
        expected_arch: "qwen35",
        requires_hadamard: true,
    },
    // Previous-generation (non-Bonsai-2) 27B family: `qwen35` hybrids with
    // no Hadamard fold.
    ManifestEntry {
        name: "bonsai-27b-q1_0",
        repo: Repo::Fixed("prism-ml/Bonsai-27B-gguf"),
        file: "Bonsai-27B-Q1_0.gguf",
        size: 3_803_452_480,
        sha256: "17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0",
        alternate_sha256: &[],
        expected_arch: "qwen35",
        requires_hadamard: false,
    },
    ManifestEntry {
        name: "ternary-bonsai-27b-pq2_0",
        repo: Repo::Fixed("prism-ml/Ternary-Bonsai-27B-gguf"),
        file: "Ternary-Bonsai-27B-PQ2_0.gguf",
        size: 7_165_121_600,
        sha256: "e4781999f1997ef97ce0c58d05750835acc999d18d83ee6489ba7ac7b14cb5f6",
        alternate_sha256: &[],
        expected_arch: "qwen35",
        requires_hadamard: false,
    },
    ManifestEntry {
        name: "ternary-bonsai-27b-q2_0",
        repo: Repo::Fixed("prism-ml/Ternary-Bonsai-27B-gguf"),
        file: "Ternary-Bonsai-27B-Q2_0.gguf",
        size: 7_165_121_600,
        sha256: "868c11714cf8fe47f5ec9eeb2be0ab1a337112886f92ee0ede6b855c4fa31757",
        alternate_sha256: &[],
        expected_arch: "qwen35",
        requires_hadamard: false,
    },
    // The current upstream object's digest is primary; the digest this
    // project's legacy goldens were captured on is a warned alternate.
    ManifestEntry {
        name: "bonsai-8b",
        repo: Repo::Fixed("prism-ml/Bonsai-8B-gguf"),
        file: "Bonsai-8B.gguf",
        size: 1_158_654_496,
        sha256: "284a335aa3fb2ced3b1b01fcb40b08aa783e3b70832767f0dd2e3fdfa134bd54",
        alternate_sha256: &["ead25897bc034fa52569d0c6d054ce38216f95db09900c8add8f6bbfb370cff1"],
        expected_arch: "qwen3",
        requires_hadamard: false,
    },
];

/// Resolve `bonsai2-27b` (design §5.8's `oxibonsai pull bonsai2-27b
/// [--band ptq1|pq2] [--vision]`) to named entries; PQ2_0 is the default
/// band.
fn resolve_bonsai2_27b_alias(band: &str, vision: bool) -> anyhow::Result<Vec<&'static str>> {
    let primary = match band {
        "pq2" | "pq2_0" => "bonsai2-27b-pq2_0",
        "ptq1" | "ptq1_0" => "bonsai2-27b-ptq1_0",
        other => anyhow::bail!("invalid --band '{other}': expected ptq1 or pq2"),
    };
    let mut names = vec![primary];
    if vision {
        names.push("bonsai2-27b-mmproj");
    }
    Ok(names)
}

pub(crate) fn find_manifest_entry(name: &str) -> Option<&'static ManifestEntry> {
    let lower = name.to_ascii_lowercase();
    MANIFEST.iter().find(|e| e.name == lower)
}

/// One resolved download: a URL plus its manifest entry (`None` for a bare
/// URL).
#[derive(Debug)]
pub(crate) struct DownloadJob {
    url: String,
    file_name: String,
    entry: Option<&'static ManifestEntry>,
}

fn resolve_jobs(model_or_url: &str, band: &str, vision: bool) -> anyhow::Result<Vec<DownloadJob>> {
    resolve_jobs_with(model_or_url, band, vision, &hf_base_url(), &|repo: Repo| {
        repo.resolve()
    })
}

/// [`resolve_jobs`] with the base URL and repository resolution passed in
/// (pure — the unit tests need no environment).
fn resolve_jobs_with(
    model_or_url: &str,
    band: &str,
    vision: bool,
    base_url: &str,
    resolve_repo: &dyn Fn(Repo) -> String,
) -> anyhow::Result<Vec<DownloadJob>> {
    if model_or_url.starts_with("http://") || model_or_url.starts_with("https://") {
        let file_name = model_or_url
            .rsplit('/')
            .next()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                anyhow::anyhow!("could not derive a file name from URL '{model_or_url}'")
            })?
            .to_string();
        return Ok(vec![DownloadJob {
            url: model_or_url.to_string(),
            file_name,
            entry: None,
        }]);
    }

    let names: Vec<&'static str> = if model_or_url.eq_ignore_ascii_case("bonsai2-27b") {
        resolve_bonsai2_27b_alias(band, vision)?
    } else {
        let entry = find_manifest_entry(model_or_url).ok_or_else(|| {
            let known: Vec<&str> = MANIFEST.iter().map(|e| e.name).collect();
            anyhow::anyhow!(
                "unknown model '{model_or_url}': pass a known name ({}, or 'bonsai2-27b'), or a \
                 full http(s) URL",
                known.join(", ")
            )
        })?;
        vec![entry.name]
    };

    names
        .into_iter()
        .map(|name| {
            let entry = find_manifest_entry(name).ok_or_else(|| {
                anyhow::anyhow!("internal error: alias target '{name}' is not in the manifest")
            })?;
            Ok(DownloadJob {
                url: hf_resolve_url(base_url, &resolve_repo(entry.repo), entry.file),
                file_name: entry.file.to_string(),
                entry: Some(entry),
            })
        })
        .collect()
}

/// `OXIBONSAI_HF_BASE_URL` (a mirror, or a local test server), else the
/// public endpoint. Deliberately NOT `HF_ENDPOINT`/`OXI_HF_ENDPOINT`: a
/// separate, Rust-native downloader must not silently inherit an operator's
/// unrelated HF client configuration.
fn hf_base_url() -> String {
    std::env::var("OXIBONSAI_HF_BASE_URL")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| HF_BASE_URL.to_string())
}

/// `<base>/<repo>/resolve/main/<file>` (pure).
fn hf_resolve_url(base_url: &str, repo: &str, file: &str) -> String {
    format!(
        "{}/{repo}/resolve/main/{file}",
        base_url.trim_end_matches('/')
    )
}

/// The optional checksum manifest: `OXIBONSAI_CHECKSUMS_FILE`, else the
/// CWD-relative `scripts/checksums.sha256` (the same resolution `oxibonsai
/// serve` uses).
fn checksums_file_path() -> PathBuf {
    std::env::var("OXIBONSAI_CHECKSUMS_FILE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("scripts/checksums.sha256"))
}

pub(crate) struct PullArgs {
    pub(crate) model_or_url: String,
    pub(crate) out_dir: String,
    pub(crate) band: String,
    pub(crate) vision: bool,
    pub(crate) force: bool,
}

pub(crate) async fn run(args: PullArgs) -> anyhow::Result<()> {
    let PullArgs {
        model_or_url,
        out_dir,
        band,
        vision,
        force,
    } = args;

    let jobs = resolve_jobs(&model_or_url, &band, vision)?;
    std::fs::create_dir_all(&out_dir)
        .map_err(|e| anyhow::anyhow!("failed to create output directory '{out_dir}': {e}"))?;

    let client = oxihttp::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(30))
        .with_webpki_roots()
        .build_https()
        .map_err(|e| anyhow::anyhow!("failed to build HTTP client: {e}"))?;

    let checksums = checksums_file_path();
    for job in jobs {
        download_one(&client, &job, Path::new(&out_dir), force, &checksums).await?;
    }
    Ok(())
}

/// A progress line on stderr, refreshed at most every
/// [`Progress::REFRESH`].
struct Progress {
    started: std::time::Instant,
    last_print: Option<std::time::Instant>,
    resumed_from: u64,
    total: Option<u64>,
}

impl Progress {
    const REFRESH: std::time::Duration = std::time::Duration::from_millis(250);

    fn new(resumed_from: u64, total: Option<u64>) -> Self {
        Self {
            started: std::time::Instant::now(),
            last_print: None,
            resumed_from,
            total,
        }
    }

    fn update(&mut self, written: u64, force: bool) {
        let now = std::time::Instant::now();
        if !force
            && self
                .last_print
                .is_some_and(|last| now - last < Self::REFRESH)
        {
            return;
        }
        self.last_print = Some(now);
        eprint!(
            "\r{}",
            progress_line(written, self.total, self.rate(written, now))
        );
        let _ = std::io::stderr().flush();
    }

    fn rate(&self, written: u64, now: std::time::Instant) -> f64 {
        let secs = (now - self.started).as_secs_f64();
        if secs > 0.0 {
            written.saturating_sub(self.resumed_from) as f64 / secs
        } else {
            0.0
        }
    }
}

/// `"  1.23 GiB / 6.71 GiB (18.3%) at 45.20 MiB/s"` (pure).
fn progress_line(written: u64, total: Option<u64>, bytes_per_sec: f64) -> String {
    let rate = format!("{}/s", format_bytes(bytes_per_sec as u64));
    match total {
        Some(total) if total > 0 => {
            let pct = (written as f64 / total as f64 * 100.0).min(100.0);
            format!(
                "  {} / {} ({pct:.1}%) at {rate}   ",
                format_bytes(written),
                format_bytes(total)
            )
        }
        _ => format!("  {} at {rate}   ", format_bytes(written)),
    }
}

fn format_bytes(n: u64) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = n as f64;
    let mut unit = 0usize;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{n} {}", UNITS[0])
    } else {
        format!("{value:.2} {}", UNITS[unit])
    }
}

/// Parse `Content-Range: bytes START-END/TOTAL` (TOTAL may be `*`) into
/// `(start, total)`.
fn parse_content_range(value: &str) -> Option<(u64, Option<u64>)> {
    let rest = value.trim().strip_prefix("bytes ")?;
    let (range, total) = rest.split_once('/')?;
    let (start, _end) = range.split_once('-')?;
    let start = start.trim().parse::<u64>().ok()?;
    let total = match total.trim() {
        "*" => None,
        t => Some(t.parse::<u64>().ok()?),
    };
    Some((start, total))
}

/// How one transfer attempt ended.
enum Leg {
    /// The body ended normally.
    Complete,
    /// The connection failed mid-body (what was written is kept).
    Interrupted(String),
}

async fn download_one(
    client: &oxihttp::HttpsClient,
    job: &DownloadJob,
    out_dir: &Path,
    force: bool,
    checksums: &Path,
) -> anyhow::Result<()> {
    let dest = out_dir.join(&job.file_name);
    let part = out_dir.join(format!("{}.part", job.file_name));

    if dest.exists() {
        if !force {
            anyhow::bail!(
                "{} already exists — pass --force to overwrite, or delete it and rerun to resume \
                 a partial download",
                dest.display()
            );
        }
        eprintln!("{}: --force given, overwriting", dest.display());
        std::fs::remove_file(&dest)?;
    }

    let mut resume_from = if part.exists() {
        std::fs::metadata(&part)?.len()
    } else {
        0
    };
    if resume_from > 0 {
        eprintln!(
            "{}: partial download found ({resume_from} bytes) — resuming",
            part.display()
        );
    }

    eprintln!("Downloading {} -> {}", job.url, dest.display());
    let mut stalled = 0u32;
    loop {
        let mut request = client
            .get(&job.url)
            .map_err(|e| anyhow::anyhow!("failed to build request for {}: {e}", job.url))?;
        if resume_from > 0 {
            request = request
                .header("Range", &format!("bytes={resume_from}-"))
                .map_err(|e| anyhow::anyhow!("failed to set Range header: {e}"))?;
        }
        let response = request
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("download request failed: {e}"))?;
        let status = response.status();
        if !status.is_success() {
            anyhow::bail!("download failed: HTTP {status} fetching {}", job.url);
        }

        // A resumed leg is accepted only as a 206 whose Content-Range
        // starts exactly where the `.part` file ends; anything else
        // restarts from byte 0 instead of splicing mismatched bytes.
        let mut append = false;
        let mut total = None;
        if resume_from > 0 {
            let range = response
                .header("content-range")
                .and_then(parse_content_range);
            match (status.as_u16(), range) {
                (206, Some((start, range_total))) if start == resume_from => {
                    append = true;
                    total = range_total;
                }
                (206, other) => {
                    eprintln!(
                        "  server answered 206 with Content-Range start {:?} for a resume at \
                         byte {resume_from}; restarting from the beginning",
                        other.map(|(start, _)| start)
                    );
                    std::fs::remove_file(&part).ok();
                    resume_from = 0;
                    stalled += 1;
                    if stalled > MAX_STALLED_ATTEMPTS {
                        anyhow::bail!("gave up: the server keeps answering a mismatched range");
                    }
                    continue;
                }
                _ => {
                    eprintln!(
                        "  server does not support resume for this URL; restarting from the \
                         beginning"
                    );
                    resume_from = 0;
                    std::fs::remove_file(&part).ok();
                }
            }
        }
        let total = total
            .or_else(|| response.content_length().map(|len| len + resume_from))
            .or(job.entry.map(|e| e.size));

        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .append(append)
            .truncate(!append)
            .open(&part)
            .map_err(|e| anyhow::anyhow!("failed to open '{}': {e}", part.display()))?;

        let mut written = resume_from;
        let mut progress = Progress::new(resume_from, total);
        progress.update(written, true);
        let leg = stream_body_to_file(response, &mut file, &mut written, &mut progress).await?;
        file.flush()?;
        progress.update(written, true);
        eprintln!();

        let incomplete = total.is_some_and(|t| written < t);
        match (&leg, incomplete) {
            (Leg::Complete, false) => break,
            (leg, _) => {
                let reason = match leg {
                    Leg::Interrupted(e) => e.clone(),
                    Leg::Complete => "the connection closed early".to_string(),
                };
                if written > resume_from {
                    stalled = 0;
                } else {
                    stalled += 1;
                }
                if stalled > MAX_STALLED_ATTEMPTS {
                    anyhow::bail!(
                        "download stalled {MAX_STALLED_ATTEMPTS} times in a row at byte \
                         {written} ({reason}); rerun `oxibonsai pull` to resume"
                    );
                }
                eprintln!("  transfer interrupted at byte {written} ({reason}) — resuming");
                resume_from = written;
            }
        }
    }

    if let Err(e) = verify(&part, job, checksums) {
        // A refused file must never linger as a `.part` a later run would
        // "resume" (and so accept), nor appear at `dest`.
        std::fs::remove_file(&part).ok();
        return Err(e);
    }
    std::fs::rename(&part, &dest)
        .map_err(|e| anyhow::anyhow!("failed to finalize '{}': {e}", dest.display()))?;
    eprintln!("Done: {}", dest.display());
    Ok(())
}

/// Stream the response body into `file` frame by frame (constant memory):
/// `oxihttp`'s `Response::body_stream()` wrapped as an `axum::body::Body`,
/// whose `http_body::Body::poll_frame` drives it with no extra dependency.
async fn stream_body_to_file(
    response: oxihttp::Response,
    file: &mut std::fs::File,
    written: &mut u64,
    progress: &mut Progress,
) -> anyhow::Result<Leg> {
    use axum::body::HttpBody as _;
    let mut body = axum::body::Body::from_stream(response.body_stream());
    loop {
        let frame = std::future::poll_fn(|cx| std::pin::Pin::new(&mut body).poll_frame(cx)).await;
        match frame {
            None => return Ok(Leg::Complete),
            Some(Err(e)) => return Ok(Leg::Interrupted(e.to_string())),
            Some(Ok(frame)) => {
                if let Ok(data) = frame.into_data() {
                    file.write_all(&data)
                        .map_err(|e| anyhow::anyhow!("failed to write the download: {e}"))?;
                    *written += data.len() as u64;
                    progress.update(*written, false);
                }
            }
        }
    }
}

/// sec-12: verify a downloaded (still-`.part`) file. See the module docs for
/// the order and the fail-closed rules.
fn verify(part_path: &Path, job: &DownloadJob, checksums: &Path) -> anyhow::Result<()> {
    let checksum_text = std::fs::read_to_string(checksums).ok();
    match job.entry {
        Some(entry) => verify_named(part_path, entry, checksum_text.as_deref(), checksums),
        None => verify_bare_url(
            part_path,
            &job.file_name,
            checksum_text.as_deref(),
            checksums,
        ),
    }
}

/// Named entry: structure → exact size → embedded SHA-256 (fail-closed) →
/// cross-check against the checksum manifest when it lists the file.
pub(crate) fn verify_named(
    part_path: &Path,
    entry: &ManifestEntry,
    checksum_text: Option<&str>,
    checksums_path: &Path,
) -> anyhow::Result<()> {
    verify_gguf_structure(
        part_path,
        Some(entry.expected_arch),
        entry.requires_hadamard,
    )?;

    let actual_size = std::fs::metadata(part_path)?.len();
    if actual_size != entry.size {
        anyhow::bail!(
            "size MISMATCH for {}: expected {} bytes, got {actual_size} — refusing a truncated, \
             padded or different file",
            entry.file,
            entry.size
        );
    }

    eprintln!("Verifying SHA-256 of {}...", entry.file);
    let actual = sha256_hex(part_path)?;
    if actual.eq_ignore_ascii_case(entry.sha256) {
        eprintln!("  checksum OK (upstream {})", entry.sha256);
    } else if entry
        .alternate_sha256
        .iter()
        .any(|alt| alt.eq_ignore_ascii_case(&actual))
    {
        eprintln!(
            "  WARNING: {} matches a previously known-good digest ({actual}), not the current \
             upstream object ({}): upstream replaced this file after this project's goldens \
             were captured. Accepting the known-good version.",
            entry.file, entry.sha256
        );
    } else {
        anyhow::bail!(
            "checksum MISMATCH for {}: expected {} (the upstream object compiled into this \
             binary), got {actual} — refusing a corrupted or tampered download",
            entry.file,
            entry.sha256
        );
    }

    if let Some(text) = checksum_text {
        if let Some(listed) =
            oxibonsai_runtime::serve_shared::lookup_expected_checksum(text, Path::new(entry.file))
        {
            let known = listed.eq_ignore_ascii_case(entry.sha256)
                || entry
                    .alternate_sha256
                    .iter()
                    .any(|alt| alt.eq_ignore_ascii_case(&listed));
            if listed.eq_ignore_ascii_case(&actual) {
                eprintln!("  cross-check OK against {}", checksums_path.display());
            } else if known {
                eprintln!(
                    "  NOTE: {} records the other known-good version of {} ({listed})",
                    checksums_path.display(),
                    entry.file
                );
            } else {
                anyhow::bail!(
                    "checksum authority conflict for {}: {} records {listed}, which matches \
                     neither the download ({actual}) nor any digest this binary knows — \
                     refusing until the manifest is corrected",
                    entry.file,
                    checksums_path.display()
                );
            }
        }
    }
    Ok(())
}

/// Bare URL: structure (magic + a declared architecture) → SHA-256 against
/// the checksum manifest when it lists the file name.
fn verify_bare_url(
    part_path: &Path,
    file_name: &str,
    checksum_text: Option<&str>,
    checksums_path: &Path,
) -> anyhow::Result<()> {
    verify_gguf_structure(part_path, None, false)?;
    let listed = checksum_text.and_then(|text| {
        oxibonsai_runtime::serve_shared::lookup_expected_checksum(text, Path::new(file_name))
    });
    match listed {
        Some(expected) => {
            let actual = sha256_hex(part_path)?;
            if !actual.eq_ignore_ascii_case(&expected) {
                anyhow::bail!(
                    "checksum MISMATCH for {file_name}: {} records {expected}, got {actual} — \
                     refusing a corrupted or tampered download",
                    checksums_path.display()
                );
            }
            eprintln!("  checksum OK against {}", checksums_path.display());
        }
        None => eprintln!(
            "  NOTE: {file_name} is not a named entry and {} lists no digest for it; only the \
             GGUF structure was verified",
            checksums_path.display()
        ),
    }
    Ok(())
}

fn sha256_hex(path: &Path) -> anyhow::Result<String> {
    oxibonsai_runtime::serve_shared::compute_sha256_hex(path)
        .map_err(|e| anyhow::anyhow!("failed to hash '{}': {e}", path.display()))?
        .ok_or_else(|| anyhow::anyhow!("failed to compute a digest for '{}'", path.display()))
}

/// The structural check (design §5.8): GGUF magic (a successful parse),
/// `general.architecture` present — and equal to `expected_arch` when
/// given — and, when `requires_hadamard`, `prism.hadamard.version == 1`.
/// Reads only the metadata through the memory map.
pub(crate) fn verify_gguf_structure(
    path: &Path,
    expected_arch: Option<&str>,
    requires_hadamard: bool,
) -> anyhow::Result<()> {
    let mmap = oxibonsai_core::gguf::reader::mmap_gguf_file(path).map_err(|e| {
        anyhow::anyhow!(
            "failed to open '{}' for structural verification: {e}",
            path.display()
        )
    })?;
    let gguf = oxibonsai_core::gguf::reader::GgufFile::parse(&mmap).map_err(|e| {
        anyhow::anyhow!(
            "downloaded file failed GGUF structural verification (bad magic, or corrupted): {e}"
        )
    })?;
    let arch = gguf
        .metadata
        .get_string(oxibonsai_core::gguf::tensor_info::keys::GENERAL_ARCHITECTURE)
        .map_err(|_| {
            anyhow::anyhow!(
                "downloaded GGUF has no general.architecture key — not a recognizable model file"
            )
        })?;
    if let Some(expected) = expected_arch {
        if arch != expected {
            anyhow::bail!(
                "downloaded file declares general.architecture = '{arch}', expected '{expected}' \
                 — refusing the wrong model file"
            );
        }
    }
    eprintln!("  structural check: GGUF magic OK, general.architecture = '{arch}'");

    if requires_hadamard {
        let version = gguf
            .metadata
            .get_u32("prism.hadamard.version")
            .map_err(|_| {
                anyhow::anyhow!(
                    "downloaded file is missing prism.hadamard.version — expected a Bonsai 2 \
                 Hadamard-rotated artifact"
                )
            })?;
        if version != 1 {
            anyhow::bail!(
                "downloaded file declares prism.hadamard.version = {version}, expected 1 — this \
                 build only understands the version-1 Hadamard contract"
            );
        }
        eprintln!("  structural check: prism.hadamard.version = 1 OK");
    }
    Ok(())
}

#[cfg(test)]
#[path = "pull_tests.rs"]
mod tests;
