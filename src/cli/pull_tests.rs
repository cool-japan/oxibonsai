//! Unit tests for `pull.rs` (sibling file, declared there via `#[path]`, so
//! `super` still names that module). Everything here is pure except the one
//! test of the env-reading wrapper, which holds the crate-wide env lock and
//! restores through RAII guards.

use super::*;
use crate::cli::util::test_env::{self, EnvVarGuard};
use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue, TensorEntry, TensorType};

// ── manifest + job resolution ───────────────────────────────────────────────

#[test]
fn find_manifest_entry_is_case_insensitive() {
    assert!(find_manifest_entry("BONSAI2-27B-PQ2_0").is_some());
    assert!(find_manifest_entry("bonsai2-27b-pq2_0").is_some());
    assert!(find_manifest_entry("does-not-exist").is_none());
}

#[test]
fn resolve_bonsai2_27b_alias_defaults_and_band() {
    assert_eq!(
        resolve_bonsai2_27b_alias("pq2", false).expect("valid"),
        vec!["bonsai2-27b-pq2_0"]
    );
    assert_eq!(
        resolve_bonsai2_27b_alias("ptq1", false).expect("valid"),
        vec!["bonsai2-27b-ptq1_0"]
    );
}

#[test]
fn resolve_bonsai2_27b_alias_with_vision_adds_mmproj() {
    assert_eq!(
        resolve_bonsai2_27b_alias("pq2", true).expect("valid"),
        vec!["bonsai2-27b-pq2_0", "bonsai2-27b-mmproj"]
    );
    assert_eq!(
        resolve_bonsai2_27b_alias("ptq1", true).expect("valid"),
        vec!["bonsai2-27b-ptq1_0", "bonsai2-27b-mmproj"]
    );
}

#[test]
fn resolve_bonsai2_27b_alias_rejects_unknown_band() {
    let err = resolve_bonsai2_27b_alias("q8", false).expect_err("unknown band");
    assert!(err.to_string().contains("q8"), "{err}");
}

fn default_repos(repo: Repo) -> String {
    repo.resolve_with(None, None)
}

#[test]
fn resolve_jobs_treats_a_url_as_a_direct_download() {
    let jobs = resolve_jobs_with(
        "https://example.com/foo/Model.gguf",
        "pq2",
        false,
        HF_BASE_URL,
        &default_repos,
    )
    .expect("must resolve a bare URL");
    assert_eq!(jobs.len(), 1);
    assert_eq!(jobs[0].file_name, "Model.gguf");
    assert!(jobs[0].entry.is_none());
}

#[test]
fn resolve_jobs_rejects_an_unknown_name() {
    let err = resolve_jobs_with(
        "not-a-real-model",
        "pq2",
        false,
        HF_BASE_URL,
        &default_repos,
    )
    .expect_err("must reject");
    let msg = err.to_string();
    assert!(msg.contains("unknown model"), "{msg}");
    assert!(msg.contains("bonsai2-27b-pq2_0"), "{msg}");
}

#[test]
fn resolve_jobs_resolves_a_named_manifest_entry_to_its_hf_url() {
    let jobs = resolve_jobs_with("bonsai-8b", "pq2", false, HF_BASE_URL, &default_repos)
        .expect("known name");
    assert_eq!(
        jobs[0].url,
        "https://huggingface.co/prism-ml/Bonsai-8B-gguf/resolve/main/Bonsai-8B.gguf"
    );
    let dev = resolve_jobs_with(
        "bonsai2-27b-q2_0",
        "pq2",
        false,
        "http://mirror/",
        &default_repos,
    )
    .expect("dev entry");
    assert_eq!(
        dev[0].url,
        "http://mirror/prism-ml/Ternary-Bonsai-2-27B-gguf-dev/resolve/main/\
         Ternary-Bonsai-2-27B-Q2_0-prism-fork-required.gguf"
    );
    let both =
        resolve_jobs_with("bonsai2-27b", "ptq1", true, HF_BASE_URL, &default_repos).expect("alias");
    assert_eq!(both.len(), 2);
    assert_eq!(both[0].file_name, "Ternary-Bonsai-2-27B-PTQ1_0.gguf");
    assert_eq!(both[1].file_name, "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf");
}

#[test]
fn repo_overrides_apply_to_the_two_bonsai2_repositories_only() {
    let primary = Repo::Bonsai2.resolve_with(Some("owner/primary"), Some("owner/dev"));
    let dev = Repo::Bonsai2Dev.resolve_with(Some("owner/primary"), Some("owner/dev"));
    let fixed = Repo::Fixed("prism-ml/Bonsai-8B-gguf").resolve_with(Some("x/y"), Some("z/w"));
    assert_eq!(primary, "owner/primary");
    assert_eq!(dev, "owner/dev");
    assert_eq!(fixed, "prism-ml/Bonsai-8B-gguf");
}

/// The env-reading wrapper (`OXI_BONSAI2_REPO` / `OXI_BONSAI2_DEV_REPO` /
/// `OXIBONSAI_HF_BASE_URL`), under the crate-wide env lock with RAII
/// restore.
#[test]
fn environment_overrides_reach_the_resolved_url() {
    let _env = test_env::lock();
    let _base = EnvVarGuard::set("OXIBONSAI_HF_BASE_URL", "http://127.0.0.1:9");
    let _primary = EnvVarGuard::set("OXI_BONSAI2_REPO", "owner/primary");
    let _dev = EnvVarGuard::set("OXI_BONSAI2_DEV_REPO", "owner/dev");
    let jobs = resolve_jobs("bonsai2-27b-pq2_0", "pq2", false).expect("resolve");
    assert_eq!(
        jobs[0].url,
        "http://127.0.0.1:9/owner/primary/resolve/main/Ternary-Bonsai-2-27B-PQ2_0.gguf"
    );
    let jobs = resolve_jobs("bonsai2-27b-q2_0", "pq2", false).expect("resolve");
    assert!(
        jobs[0].url.contains("/owner/dev/resolve/main/"),
        "{}",
        jobs[0].url
    );
}

#[test]
fn hf_resolve_url_uses_the_standard_hf_layout_by_default() {
    assert_eq!(
        hf_resolve_url(HF_BASE_URL, "prism-ml/Bonsai-8B-gguf", "Bonsai-8B.gguf"),
        "https://huggingface.co/prism-ml/Bonsai-8B-gguf/resolve/main/Bonsai-8B.gguf"
    );
    // A trailing slash on a mirror base is tolerated.
    assert_eq!(
        hf_resolve_url("http://mirror/", "o/r", "f.gguf"),
        "http://mirror/o/r/resolve/main/f.gguf"
    );
    // With no override in the environment the public endpoint is used.
    let _env = test_env::lock();
    let _base = EnvVarGuard::remove("OXIBONSAI_HF_BASE_URL");
    assert_eq!(hf_base_url(), HF_BASE_URL);
    let _empty = EnvVarGuard::set("OXIBONSAI_HF_BASE_URL", "");
    assert_eq!(hf_base_url(), HF_BASE_URL, "an empty override is ignored");
}

#[test]
fn manifest_entries_are_well_formed() {
    let mut seen = std::collections::HashSet::new();
    for entry in MANIFEST {
        assert!(
            seen.insert(entry.name),
            "duplicate manifest name: {}",
            entry.name
        );
        assert!(entry.size > 0, "{}", entry.name);
        for digest in std::iter::once(&entry.sha256).chain(entry.alternate_sha256) {
            assert_eq!(digest.len(), 64, "{}: {digest}", entry.name);
            assert!(
                digest.chars().all(|c| c.is_ascii_hexdigit()),
                "{}: {digest}",
                entry.name
            );
        }
        assert!(!entry.expected_arch.is_empty(), "{}", entry.name);
    }
    // Every Hadamard-folded Bonsai 2 language file requires the contract
    // (incl. the dev-repo Q2_0, which declares prism.hadamard.version = 1);
    // the projector and the previous generation do not.
    for name in [
        "bonsai2-27b-ptq1_0",
        "bonsai2-27b-pq2_0",
        "bonsai2-27b-q2_0",
    ] {
        let entry = find_manifest_entry(name).expect("entry");
        assert!(entry.requires_hadamard, "{name}");
        assert_eq!(entry.expected_arch, "qwen35", "{name}");
    }
    let mmproj = find_manifest_entry("bonsai2-27b-mmproj").expect("mmproj");
    assert_eq!(mmproj.expected_arch, "clip");
    assert!(!mmproj.requires_hadamard);
    for name in [
        "bonsai-27b-q1_0",
        "ternary-bonsai-27b-pq2_0",
        "ternary-bonsai-27b-q2_0",
    ] {
        let entry = find_manifest_entry(name).expect("entry");
        assert!(!entry.requires_hadamard, "{name}");
        assert_eq!(entry.expected_arch, "qwen35", "{name}");
    }
    let bonsai_8b = find_manifest_entry("bonsai-8b").expect("bonsai-8b");
    assert_eq!(bonsai_8b.expected_arch, "qwen3");
    assert_eq!(
        bonsai_8b.alternate_sha256,
        &["ead25897bc034fa52569d0c6d054ce38216f95db09900c8add8f6bbfb370cff1"]
    );
}

/// Drift test: every digest the repository's own `scripts/checksums.sha256`
/// records for a manifest file must be that entry's primary digest or one
/// of its declared alternates (read through `CARGO_MANIFEST_DIR`, never the
/// CWD).
#[test]
fn pull_manifest_agrees_with_the_repo_checksums_file() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/checksums.sha256");
    let text = std::fs::read_to_string(&path).expect("the repository ships its checksums file");
    let mut cross_checked = 0usize;
    for entry in MANIFEST {
        if let Some(listed) =
            oxibonsai_runtime::serve_shared::lookup_expected_checksum(&text, Path::new(entry.file))
        {
            let known = listed == entry.sha256 || entry.alternate_sha256.contains(&listed.as_str());
            assert!(
                known,
                "{} is recorded as {listed} in scripts/checksums.sha256, which is neither the \
                 manifest digest {} nor a declared alternate",
                entry.file, entry.sha256
            );
            cross_checked += 1;
        }
    }
    // PTQ1_0, PQ2_0, the mmproj and Bonsai-8B are all recorded there.
    assert!(
        cross_checked >= 4,
        "only {cross_checked} entries cross-checked"
    );
}

// ── Content-Range + progress formatting ─────────────────────────────────────

#[test]
fn parse_content_range_reads_start_and_total() {
    assert_eq!(
        parse_content_range("bytes 100-199/200"),
        Some((100, Some(200)))
    );
    assert_eq!(parse_content_range("bytes 0-99/*"), Some((0, None)));
    assert_eq!(parse_content_range("items 0-1/2"), None);
    assert_eq!(parse_content_range("bytes x-1/2"), None);
}

#[test]
fn progress_line_reports_bytes_percentage_and_rate() {
    let line = progress_line(
        512 * 1024 * 1024,
        Some(1024 * 1024 * 1024),
        2.0 * 1024.0 * 1024.0,
    );
    assert!(line.contains("512.00 MiB / 1.00 GiB (50.0%)"), "{line}");
    assert!(line.contains("2.00 MiB/s"), "{line}");
    let unknown = progress_line(1536, None, 0.0);
    assert!(unknown.contains("1.50 KiB"), "{unknown}");
}

#[test]
fn format_bytes_renders_human_units() {
    assert_eq!(format_bytes(500), "500 B");
    assert_eq!(format_bytes(1536), "1.50 KiB");
    assert_eq!(format_bytes(7_206_168_928), "6.71 GiB");
}

// ── fail-closed verification of a named entry ───────────────────────────────

fn gguf_bytes(arch: &str, hadamard_version: Option<u32>) -> Vec<u8> {
    let mut w = GgufWriter::new();
    w.add_metadata(
        "general.architecture",
        MetadataWriteValue::Str(arch.to_string()),
    );
    if let Some(version) = hadamard_version {
        w.add_metadata("prism.hadamard.version", MetadataWriteValue::U32(version));
    }
    w.add_tensor(TensorEntry {
        name: "output_norm.weight".to_string(),
        shape: vec![8],
        tensor_type: TensorType::F32,
        data: vec![0u8; 32],
    });
    w.to_bytes().expect("serialize fixture")
}

struct Written {
    dir: PathBuf,
    path: PathBuf,
    sha256: &'static str,
    size: u64,
}

impl Drop for Written {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn write_fixture(tag: &str, bytes: &[u8]) -> Written {
    let dir = crate::cli::test_fixtures::scratch_dir(&format!("pull_verify_{tag}"));
    let path = dir.join("Model.gguf.part");
    std::fs::write(&path, bytes).expect("write fixture");
    let digest = sha256_hex(&path).expect("hash");
    Written {
        dir,
        path,
        sha256: Box::leak(digest.into_boxed_str()),
        size: bytes.len() as u64,
    }
}

fn entry_for(fixture: &Written, arch: &'static str, hadamard: bool) -> ManifestEntry {
    ManifestEntry {
        name: "test-entry",
        repo: Repo::Fixed("test/repo"),
        file: "Model.gguf",
        size: fixture.size,
        sha256: fixture.sha256,
        alternate_sha256: &[],
        expected_arch: arch,
        requires_hadamard: hadamard,
    }
}

const NO_CHECKSUMS: Option<&str> = None;

fn no_checksums_path() -> PathBuf {
    std::env::temp_dir().join("oxibonsai_pull_tests_no_checksums_file")
}

#[test]
fn a_matching_named_download_is_accepted() {
    let fix = write_fixture("ok", &gguf_bytes("qwen35", Some(1)));
    verify_named(
        &fix.path,
        &entry_for(&fix, "qwen35", true),
        NO_CHECKSUMS,
        &no_checksums_path(),
    )
    .expect("structure, size and digest all match");
}

#[test]
fn a_wrong_architecture_is_refused_before_anything_else() {
    let fix = write_fixture("arch", &gguf_bytes("qwen3", None));
    let err = verify_named(
        &fix.path,
        &entry_for(&fix, "qwen35", false),
        NO_CHECKSUMS,
        &no_checksums_path(),
    )
    .expect_err("wrong architecture");
    assert!(err.to_string().contains("expected 'qwen35'"), "{err}");
}

#[test]
fn a_missing_or_wrong_hadamard_version_is_refused() {
    let fix = write_fixture("no_hadamard", &gguf_bytes("qwen35", None));
    let err = verify_named(
        &fix.path,
        &entry_for(&fix, "qwen35", true),
        NO_CHECKSUMS,
        &no_checksums_path(),
    )
    .expect_err("missing contract");
    assert!(
        err.to_string().contains("missing prism.hadamard.version"),
        "{err}"
    );

    let fix = write_fixture("hadamard_v2", &gguf_bytes("qwen35", Some(2)));
    let err = verify_named(
        &fix.path,
        &entry_for(&fix, "qwen35", true),
        NO_CHECKSUMS,
        &no_checksums_path(),
    )
    .expect_err("version 2");
    assert!(
        err.to_string().contains("prism.hadamard.version = 2"),
        "{err}"
    );
}

#[test]
fn a_size_mismatch_fails_closed() {
    let fix = write_fixture("size", &gguf_bytes("qwen35", Some(1)));
    let mut entry = entry_for(&fix, "qwen35", true);
    entry.size += 1;
    let err = verify_named(&fix.path, &entry, NO_CHECKSUMS, &no_checksums_path())
        .expect_err("size mismatch");
    assert!(err.to_string().contains("size MISMATCH"), "{err}");
}

#[test]
fn a_digest_mismatch_fails_closed_with_no_checksums_file_anywhere() {
    let fix = write_fixture("digest", &gguf_bytes("qwen35", Some(1)));
    let mut entry = entry_for(&fix, "qwen35", true);
    entry.sha256 = "0000000000000000000000000000000000000000000000000000000000000000";
    let err = verify_named(&fix.path, &entry, NO_CHECKSUMS, &no_checksums_path())
        .expect_err("digest mismatch is fatal even without a checksums file");
    assert!(err.to_string().contains("checksum MISMATCH"), "{err}");
}

#[test]
fn a_known_good_alternate_digest_is_accepted() {
    let fix = write_fixture("alternate", &gguf_bytes("qwen3", None));
    let mut entry = entry_for(&fix, "qwen3", false);
    let alternates: &'static [&'static str] = Box::leak(vec![fix.sha256].into_boxed_slice());
    entry.sha256 = "1111111111111111111111111111111111111111111111111111111111111111";
    entry.alternate_sha256 = alternates;
    verify_named(&fix.path, &entry, NO_CHECKSUMS, &no_checksums_path())
        .expect("the previously known-good digest is accepted (with a warning)");
}

#[test]
fn the_checksums_file_is_an_additional_cross_checked_authority() {
    let fix = write_fixture("cross", &gguf_bytes("qwen35", Some(1)));
    let entry = entry_for(&fix, "qwen35", true);
    // Agreeing record: fine.
    let agreeing = format!("{}  models/Model.gguf\n", fix.sha256);
    verify_named(&fix.path, &entry, Some(&agreeing), &no_checksums_path()).expect("agrees");
    // A record naming a digest this binary does not know: a conflict.
    let conflicting = format!("{}  models/Model.gguf\n", "2".repeat(64));
    let err = verify_named(&fix.path, &entry, Some(&conflicting), &no_checksums_path())
        .expect_err("conflicting authority");
    assert!(
        err.to_string().contains("checksum authority conflict"),
        "{err}"
    );
    // A record of the OTHER known-good version (the Bonsai-8B situation).
    let mut with_alt = entry.clone();
    let alternates: &'static [&'static str] = Box::leak(
        vec!["3333333333333333333333333333333333333333333333333333333333333333"].into_boxed_slice(),
    );
    with_alt.alternate_sha256 = alternates;
    let other_version = format!("{}  models/Model.gguf\n", "3".repeat(64));
    verify_named(
        &fix.path,
        &with_alt,
        Some(&other_version),
        &no_checksums_path(),
    )
    .expect("the other known-good version is only a note");
}

#[test]
fn a_bare_url_is_verified_against_the_checksums_file_when_listed() {
    let fix = write_fixture("bare", &gguf_bytes("qwen3", None));
    verify_bare_url(&fix.path, "Model.gguf", NO_CHECKSUMS, &no_checksums_path())
        .expect("structure only when nothing lists it");
    let wrong = format!("{}  Model.gguf\n", "4".repeat(64));
    let err = verify_bare_url(&fix.path, "Model.gguf", Some(&wrong), &no_checksums_path())
        .expect_err("listed and wrong");
    assert!(err.to_string().contains("checksum MISMATCH"), "{err}");
}
