//! `MET-10` guard: no Metal kernel family compiles its own library, opens its
//! own device, or creates its own command queue any more.
//!
//! Before `MET-10` the K-quant, standard-quant, FP8 GEMV and FP8 batch-prefill
//! families each held a private `OnceLock` singleton that called
//! `device.new_library_with_source(..)` once **per kernel** — 16 uncached
//! MSL compilations on first use, every process start — and, before the
//! `MET-08` session split, each also opened its own `Device::system_default()`
//! and `new_command_queue()`: five independent queues on one device that no
//! engine-pool session could see. Every one of those kernels already rides the
//! combined metallib `build.rs` embeds; they now resolve their pipelines by
//! name through `MetalGraph::pipeline_for` and dispatch on the current
//! session's queue.
//!
//! This test keeps it that way by scanning the crate's own `gpu_backend`
//! sources (non-test code only, comments stripped):
//!
//! 1. the four kernel-family files contain none of `new_library_with_source(`,
//!    `Device::system_default(` or `new_command_queue(`;
//! 2. across the whole `gpu_backend` tree, the only file that compiles a
//!    library from source is `metal_graph/pipelines.rs` — the combined
//!    library's own embedded → disk-cache → `xcrun` → runtime-source cascade,
//!    plus the best-effort bf16 sidecar. The batched prefill attention kernels
//!    (`metal_prefill/attention.rs`) and the Qwen3.5 hybrid kernels
//!    (`metal_full_layer/qwen35.rs`) ride the combined metallib and resolve
//!    their pipelines through `pipeline_for`; any *new* private library fails
//!    this;
//! 3. `metal_prefill/attention.rs` does not even mention the source-compile
//!    call any more — the one-off prefill-attention compile path (its own
//!    library, a fixed pid-named temp build directory and non-atomic cache
//!    writes) is gone, not merely unreachable.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// The four kernel families `MET-10` ported onto `pipeline_for`.
const PORTED_FAMILIES: [&str; 4] = [
    "metal_fp8_kernels.rs",
    "metal_fp8_prefill.rs",
    "metal_k_quant_kernels.rs",
    "metal_q_std_kernels.rs",
];

/// The only `gpu_backend` file allowed to compile an MSL library from source.
const LIBRARY_OWNERS: [&str; 1] = ["metal_graph/pipelines.rs"];

/// Files that dispatch kernels of the combined metallib and must resolve
/// them through `pipeline_for` rather than own any Metal library, device or
/// queue of their own.
const COMBINED_LIBRARY_CLIENTS: [&str; 2] =
    ["metal_prefill/attention.rs", "metal_full_layer/qwen35.rs"];

/// Root of the scan: this crate's `src/gpu_backend`.
fn gpu_backend_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("gpu_backend")
}

/// Every `.rs` file under `dir`, recursively, as paths relative to `root`.
fn rust_sources(root: &Path, dir: &Path, out: &mut Vec<PathBuf>) {
    let entries =
        std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            rust_sources(root, &path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            let rel = path
                .strip_prefix(root)
                .expect("path under the scan root")
                .to_path_buf();
            out.push(rel);
        }
    }
}

/// Whether a file is compiled only under `cfg(test)` by naming convention
/// (`tests.rs`, `tests_*.rs`, `*_tests.rs` — how this crate names its
/// `#[cfg(test)] #[path]` siblings).
fn is_test_only_file(rel: &Path) -> bool {
    let name = rel.file_name().and_then(|n| n.to_str()).unwrap_or_default();
    name == "tests.rs" || name.starts_with("tests_") || name.ends_with("_tests.rs")
}

/// Drop a trailing `//` comment (including `///` and `//!` docs) from a line.
///
/// Good enough for this scan: none of the needles below ever appears inside
/// a string literal in non-test code, and a `//` inside a string only makes
/// the scan *stricter* (it keeps less text), never laxer.
fn strip_line_comment(line: &str) -> &str {
    match line.find("//") {
        Some(at) => &line[..at],
        None => line,
    }
}

/// The non-test, comment-free code of one source file.
///
/// A `#[cfg(test)]` / `#[cfg(all(test, ..))]` attribute followed by an inline
/// `mod name { .. }` skips that whole module by brace depth; followed by an
/// out-of-line `mod name;` it skips just that declaration.
fn non_test_code(source: &str) -> String {
    let mut out = String::new();
    let mut lines = source.lines().peekable();
    while let Some(line) = lines.next() {
        let trimmed = line.trim_start();
        let is_test_cfg = trimmed.starts_with("#[cfg(test)]")
            || trimmed.starts_with("#[cfg(all(test")
            || trimmed.starts_with("#[cfg(any(test");
        if !is_test_cfg {
            out.push_str(strip_line_comment(line));
            out.push('\n');
            continue;
        }
        // Skip any further attributes (`#[path = ..]`, `#[allow(..)]`).
        while lines
            .peek()
            .is_some_and(|next| next.trim_start().starts_with("#["))
        {
            lines.next();
        }
        let Some(item) = lines.next() else {
            break;
        };
        let item_code = strip_line_comment(item);
        if !item_code.contains('{') {
            // `mod name;` or a single-line item: nothing more to skip.
            continue;
        }
        let mut depth: i64 = 0;
        for ch in item_code.chars() {
            match ch {
                '{' => depth += 1,
                '}' => depth -= 1,
                _ => {}
            }
        }
        while depth > 0 {
            let Some(body) = lines.next() else {
                break;
            };
            for ch in strip_line_comment(body).chars() {
                match ch {
                    '{' => depth += 1,
                    '}' => depth -= 1,
                    _ => {}
                }
            }
        }
    }
    out
}

/// `(relative path, non-test code)` for every non-test-only `gpu_backend` file.
fn scanned_sources() -> Vec<(String, String)> {
    let root = gpu_backend_dir();
    let mut files = Vec::new();
    rust_sources(&root, &root, &mut files);
    files.sort();
    assert!(
        files.len() > 20,
        "the scan found only {} files under {} — the walk is broken, not the code",
        files.len(),
        root.display()
    );
    files
        .into_iter()
        .filter(|rel| !is_test_only_file(rel))
        .map(|rel| {
            let text = std::fs::read_to_string(root.join(&rel))
                .unwrap_or_else(|e| panic!("read {}: {e}", rel.display()));
            let key = rel
                .components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect::<Vec<_>>()
                .join("/");
            (key, non_test_code(&text))
        })
        .collect()
}

#[test]
fn the_scanner_strips_test_modules_and_comments() {
    let sample = "\
fn real() { device.new_library_with_source(src, &opts); }
// device.new_library_with_source(in_a_comment)
#[cfg(test)]
mod tests {
    fn inner() {
        if x { device.new_library_with_source(in_a_test); }
    }
}
#[cfg(all(test, feature = \"metal\"))]
#[path = \"x_tests.rs\"]
mod x_tests;
fn after() { device.new_command_queue(); }
";
    let code = non_test_code(sample);
    assert_eq!(
        code.matches("new_library_with_source(").count(),
        1,
        "{code}"
    );
    assert!(
        code.contains("fn after()"),
        "code after a test module must be kept: {code}"
    );
    assert!(code.contains("new_command_queue("), "{code}");
    assert!(
        !code.contains("in_a_test") && !code.contains("in_a_comment"),
        "{code}"
    );
}

#[test]
fn ported_kernel_families_own_no_library_device_or_queue() {
    let sources = scanned_sources();
    for family in PORTED_FAMILIES {
        let (_, code) = sources
            .iter()
            .find(|(path, _)| path == family)
            .unwrap_or_else(|| panic!("{family} not found under src/gpu_backend"));
        for needle in [
            "new_library_with_source(",
            "Device::system_default(",
            "new_command_queue(",
            "OnceLock",
        ] {
            assert!(
                !code.contains(needle),
                "{family} contains `{needle}` in non-test code: a kernel family must resolve \
                 its pipelines through `MetalGraph::pipeline_for` and dispatch on the current \
                 session's queue, not own Metal state (MET-10)"
            );
        }
        assert!(
            code.contains("pipeline_for("),
            "{family} no longer resolves pipelines through `pipeline_for` — where do its \
             kernels come from?"
        );
    }
}

#[test]
fn only_the_known_owners_compile_a_metal_library_from_source() {
    let compiling: BTreeSet<String> = scanned_sources()
        .into_iter()
        .filter(|(_, code)| code.contains("new_library_with_source("))
        .map(|(path, _)| path)
        .collect();
    let allowed: BTreeSet<String> = LIBRARY_OWNERS.iter().map(|s| (*s).to_string()).collect();
    let rogue: Vec<&String> = compiling.difference(&allowed).collect();
    assert!(
        rogue.is_empty(),
        "these gpu_backend files compile their own Metal library from source: {rogue:?}. \
         Add the kernel to `build.rs::ACTIVE_KERNELS` + `build_combined_msl()` and resolve it \
         with `MetalGraph::pipeline_for` instead (MET-10)"
    );
    assert!(
        compiling.contains("metal_graph/pipelines.rs"),
        "the combined library's own compile cascade disappeared from pipelines.rs — the scan \
         is not seeing the code it is meant to guard"
    );
}

#[test]
fn combined_library_clients_own_no_library_device_or_queue() {
    let sources = scanned_sources();
    for client in COMBINED_LIBRARY_CLIENTS {
        let (_, code) = sources
            .iter()
            .find(|(path, _)| path == client)
            .unwrap_or_else(|| panic!("{client} not found under src/gpu_backend"));
        for needle in [
            "new_library_with_source(",
            "new_library_with_data(",
            "Device::system_default(",
            "new_command_queue(",
        ] {
            assert!(
                !code.contains(needle),
                "{client} contains `{needle}` in non-test code: its kernels are in the combined \
                 metallib and must be resolved with `MetalGraph::pipeline_for`"
            );
        }
        assert!(
            code.contains("pipeline_for("),
            "{client} no longer resolves pipelines through `pipeline_for`"
        );
    }
}

/// The batched prefill attention module keeps no trace of a private compile
/// path: its whole text — test code and comments included — names no
/// source or data library compile, no `xcrun` build and no temp build
/// directory, because its kernels come from the combined metallib.
#[test]
fn prefill_attention_has_no_private_compile_path_left() {
    let path = gpu_backend_dir().join("metal_prefill").join("attention.rs");
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    for needle in [
        "new_library_with_source",
        "new_library_with_data",
        "xcrun",
        "temp_dir(",
    ] {
        assert!(
            !text.contains(needle),
            "{} still contains `{needle}`",
            path.display()
        );
    }
}
