//! Integration tests that parse REAL PrismML Bonsai 2 27B GGUF headers, and
//! (when available) the real dense Bonsai/Ternary-Bonsai GGUFs, proving
//! `Qwen3Config`, `HybridConfig` and `HadamardConfig` work against actual
//! file bytes rather than only synthetic fixtures (B2-02 acceptance:
//! "parses all six real 27B headers"; M-34: "each named constructor equals
//! `from_metadata` on the corresponding real file when present").
//!
//! Two independent, optional fixture sources — neither ever checked into
//! the repository nor hardcoded as an absolute path (COOLJAPAN policy:
//! "never hardcode absolute paths", "use std::env::temp_dir()/tempfile; no
//! absolute paths in code or tests"):
//!
//! - `OXIBONSAI_BONSAI2_HEADERS_DIR`: a directory of the real 27B family's
//!   `*.gguf.head` files (the first ~64 MB of each real GGUF — full magic,
//!   header, metadata and tensor-info sections, but no tensor data), staged
//!   for a validation session. These fixtures are session-scratchpad-only
//!   and never present in an ordinary checkout or CI image, so the two
//!   tests that need them are additionally `#[ignore]`d: run them with
//!   `OXIBONSAI_BONSAI2_HEADERS_DIR=<dir> cargo test -p oxibonsai-core
//!   --test config_real_headers -- --ignored`. The runtime check is kept
//!   too, as a defensive skip-not-fail for anyone who runs with `--ignored`
//!   but without the variable set.
//! - `models/*.gguf` at the workspace root (`../../models` relative to this
//!   crate, mirroring the existing `models_dir()` helper in
//!   `quant_prism_golden.rs`): the real dense Bonsai models, when a
//!   workstation happens to have them checked out locally. These are
//!   deliberately **not** `#[ignore]`d — a checkout that has the real
//!   weight files (e.g. the primary repo, post-merge) must exercise the
//!   M-34 named-constructor-vs-real-file assertions by default, since that
//!   is exactly the gate that caught B2-02's `ternary_bonsai_8b()` YaRN
//!   defect. Skipped (not failed) only when the specific file is absent.

use std::path::{Path, PathBuf};

use oxibonsai_core::config::Qwen3Config;
use oxibonsai_core::config_hybrid::HybridConfig;
use oxibonsai_core::gguf::reader::{mmap_gguf_file, GgufFile};
use oxibonsai_core::hadamard_config::HadamardConfig;
use oxibonsai_core::BonsaiError;

/// Directory holding the staged real-27B `.gguf.head` fixtures, if any.
fn real_headers_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("OXIBONSAI_BONSAI2_HEADERS_DIR")?;
    let path = PathBuf::from(dir);
    path.is_dir().then_some(path)
}

/// Workspace `models/` directory (present only on a workstation that has
/// the real weight files checked out; never shipped in the repository).
fn models_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models")
}

/// The six real 27B language-model GGUF header filenames this test suite
/// knows about (`CONTEXT.md` / `bonsai2-design.md` §0.1). The mmproj
/// (`clip` architecture) vision-tower file is exercised separately, since
/// it is expected to be *rejected*, not parsed as a language model.
const REAL_27B_HEADERS: &[&str] = &[
    "Bonsai-27B-Q1_0.gguf.head",
    "Ternary-Bonsai-2-27B-PQ2_0.gguf.head",
    "Ternary-Bonsai-2-27B-PTQ1_0.gguf.head",
    "Ternary-Bonsai-2-27B-Q2_0-prism-fork-required.gguf.head",
    "Ternary-Bonsai-27B-PQ2_0.gguf.head",
    "Ternary-Bonsai-27B-Q2_0.gguf.head",
];

/// Of [`REAL_27B_HEADERS`], the subset whose `prism.hadamard.version`
/// metadata is present (Hadamard-folded, "Bonsai 2" generation) — the other
/// three predate Hadamard folding and must yield `HadamardConfig::from_metadata
/// == Ok(None)`.
const HADAMARD_FOLDED_HEADERS: &[&str] = &[
    "Ternary-Bonsai-2-27B-PQ2_0.gguf.head",
    "Ternary-Bonsai-2-27B-PTQ1_0.gguf.head",
    "Ternary-Bonsai-2-27B-Q2_0-prism-fork-required.gguf.head",
];

const MMPROJ_HEADER: &str = "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf.head";

fn read_header(dir: &Path, name: &str) -> memmap2::Mmap {
    mmap_gguf_file(&dir.join(name))
        .unwrap_or_else(|e| panic!("failed to mmap staged fixture '{name}': {e}"))
}

#[test]
#[ignore = "requires OXIBONSAI_BONSAI2_HEADERS_DIR staged with the real 27B \
            .gguf.head fixtures; run with `-- --ignored`"]
fn all_six_real_27b_headers_parse_as_hybrid_qwen35() {
    let Some(dir) = real_headers_dir() else {
        eprintln!(
            "skipping all_six_real_27b_headers_parse_as_hybrid_qwen35: set \
             OXIBONSAI_BONSAI2_HEADERS_DIR to a directory of staged .gguf.head fixtures to run it"
        );
        return;
    };

    for name in REAL_27B_HEADERS {
        let mmap = read_header(&dir, name);
        let file = GgufFile::parse(&mmap)
            .unwrap_or_else(|e| panic!("{name}: GgufFile::parse failed: {e}"));

        // ── Qwen3Config (dense fields shared with HybridConfig::base) ──────
        let config = Qwen3Config::from_metadata_and_tensors(&file.metadata, &file.tensors)
            .unwrap_or_else(|e| {
                panic!("{name}: Qwen3Config::from_metadata_and_tensors failed: {e}")
            });
        assert_eq!(config.architecture, "qwen35", "{name}: architecture");
        assert_eq!(config.num_layers, 64, "{name}: num_layers");
        assert_eq!(config.hidden_size, 5120, "{name}: hidden_size");
        assert_eq!(config.intermediate_size, 17408, "{name}: intermediate_size");
        assert_eq!(
            config.num_attention_heads, 24,
            "{name}: num_attention_heads"
        );
        assert_eq!(config.num_kv_heads, 4, "{name}: num_kv_heads");
        assert_eq!(config.head_dim, 256, "{name}: head_dim");
        assert_eq!(config.value_length, 256, "{name}: value_length");
        assert_eq!(config.vocab_size, 248_320, "{name}: vocab_size");
        assert_eq!(
            config.max_context_length, 262_144,
            "{name}: max_context_length"
        );
        // Appendix A.4 / design table (verified by hand against a real
        // PTQ1_0 header, then confirmed identical across all six via
        // `gguf_headers_summary.txt`): every real 27B header agrees on
        // these, so they are permanent (unconditional) asserts, not just a
        // one-off spot check.
        assert_eq!(config.rope_freq_base, 1e7, "{name}: rope_freq_base");
        assert_eq!(config.rms_norm_eps, 1e-6, "{name}: rms_norm_eps");
        // M-17: none of the six real 27B headers declares
        // `qwen35.attention.sliding_window` (they use
        // `qwen35.full_attention_interval = 4` to alternate full/linear
        // layers instead), so the dense sliding-window path must stay off
        // and the forward pass must stay fully causal for this family.
        assert_eq!(
            config.sliding_window, None,
            "{name}: sliding_window (no 27B header declares the key)"
        );

        // `from_metadata` alone (no TensorStore) must resolve the same
        // vocab_size via the tokenizer.ggml.tokens fallback — proving the
        // metadata-only entry point is genuinely sufficient for every real
        // file, not just the tensor-aware one.
        let metadata_only = Qwen3Config::from_metadata(&file.metadata)
            .unwrap_or_else(|e| panic!("{name}: Qwen3Config::from_metadata failed: {e}"));
        assert_eq!(
            metadata_only.vocab_size, 248_320,
            "{name}: metadata-only vocab_size fallback"
        );

        // ── HybridConfig ────────────────────────────────────────────────────
        let hybrid = HybridConfig::from_metadata(&file.metadata)
            .unwrap_or_else(|e| panic!("{name}: HybridConfig::from_metadata failed: {e}"));
        assert_eq!(
            hybrid.full_attention_interval, 4,
            "{name}: full_attention_interval"
        );
        assert_eq!(
            hybrid.rope_dimension_count, 64,
            "{name}: rope_dimension_count"
        );
        assert_eq!(
            hybrid.rope_sections,
            [11, 11, 10, 0],
            "{name}: rope_sections"
        );
        assert_eq!(hybrid.ssm_conv_kernel, 4, "{name}: ssm_conv_kernel");
        assert_eq!(hybrid.ssm_state_size, 128, "{name}: ssm_state_size");
        assert_eq!(hybrid.ssm_group_count, 16, "{name}: ssm_group_count");
        assert_eq!(hybrid.ssm_time_step_rank, 48, "{name}: ssm_time_step_rank");
        assert_eq!(hybrid.ssm_inner_size, 6144, "{name}: ssm_inner_size");
        assert_eq!(hybrid.n_k_heads(), 16, "{name}: n_k_heads");
        assert_eq!(hybrid.n_v_heads(), 48, "{name}: n_v_heads");
        assert_eq!(hybrid.head_k_dim(), 128, "{name}: head_k_dim");
        assert_eq!(hybrid.head_v_dim(), 128, "{name}: head_v_dim");
        assert!(hybrid.validate().is_ok(), "{name}: validate()");

        let full: Vec<usize> = (0..hybrid.base.num_layers)
            .filter(|&i| hybrid.is_full_attention(i))
            .collect();
        let expected: Vec<usize> = (0..64).filter(|i| (i + 1) % 4 == 0).collect();
        assert_eq!(full, expected, "{name}: is_full_attention set");
        assert_eq!(full.len(), 16, "{name}: 16 full-attention layers");
        assert_eq!(hybrid.num_full_layers(), 16, "{name}: num_full_layers");
        assert_eq!(hybrid.num_linear_layers(), 48, "{name}: num_linear_layers");
        // `general.sampling.*` recommended defaults (design table, verified
        // real values): present and identical on every one of the six real
        // 27B headers (confirmed via `gguf_headers_summary.txt`), so these
        // are permanent asserts too, not scoped to a subset of headers the
        // way `HADAMARD_FOLDED_HEADERS` scopes the Hadamard-only fields.
        assert_eq!(hybrid.sampling_top_k, Some(20), "{name}: sampling_top_k");
        assert_eq!(hybrid.sampling_top_p, Some(0.95), "{name}: sampling_top_p");
        assert_eq!(
            hybrid.sampling_temperature,
            Some(1.0),
            "{name}: sampling_temperature"
        );

        // ── HadamardConfig ──────────────────────────────────────────────────
        let hadamard = HadamardConfig::from_metadata(&file.metadata)
            .unwrap_or_else(|e| panic!("{name}: HadamardConfig::from_metadata failed: {e}"));
        if HADAMARD_FOLDED_HEADERS.contains(name) {
            let hadamard = hadamard
                .unwrap_or_else(|| panic!("{name}: expected Some(HadamardConfig), got None"));
            assert_eq!(hadamard.block_size, 1024, "{name}: hadamard.block_size");
            assert_eq!(hadamard.folded.len(), 401, "{name}: folded.len()");
            assert!(
                hadamard.is_inverse("token_embd.weight"),
                "{name}: inverse contains token_embd.weight"
            );
            assert!(hadamard.gdn_v_grouped, "{name}: gdn_v_grouped");
            let widths = [5120usize, 6144, 17408];
            for w in widths {
                assert!(hadamard.signs_for(w).is_ok(), "{name}: signs_for({w})");
            }
            let sign_values_0_6 = hadamard.signs_for(5120).expect("width 5120 configured");
            assert_eq!(
                &sign_values_0_6[0..6],
                &[-1.0, -1.0, -1.0, 1.0, -1.0, 1.0],
                "{name}: sign_values[0..6] must match the real verified header values"
            );
            hadamard
                .validate_against_tensors(&file.tensors)
                .unwrap_or_else(|e| panic!("{name}: validate_against_tensors failed: {e}"));
        } else {
            assert!(
                hadamard.is_none(),
                "{name}: expected None (pre-Hadamard 27B generation)"
            );
        }
    }
}

#[test]
#[ignore = "requires OXIBONSAI_BONSAI2_HEADERS_DIR staged with the real 27B \
            .gguf.head fixtures; run with `-- --ignored`"]
fn mmproj_clip_architecture_is_rejected_not_silently_defaulted() {
    let Some(dir) = real_headers_dir() else {
        eprintln!(
            "skipping mmproj_clip_architecture_is_rejected_not_silently_defaulted: set \
             OXIBONSAI_BONSAI2_HEADERS_DIR"
        );
        return;
    };
    let mmap = read_header(&dir, MMPROJ_HEADER);
    let file = GgufFile::parse(&mmap).expect("mmproj header should still structurally parse");
    let err = Qwen3Config::from_metadata(&file.metadata)
        .expect_err("a `clip` architecture file must be rejected, not silently defaulted to Qwen3");
    match err {
        BonsaiError::UnsupportedArchitecture { arch } => assert_eq!(arch, "clip"),
        other => panic!("expected UnsupportedArchitecture{{arch: \"clip\"}}, got {other:?}"),
    }
    // The mmproj file has no `qwen35.*`/`prism.hadamard.*` keys at all, so
    // both hybrid-specific parsers must equally refuse it.
    assert!(HybridConfig::from_metadata(&file.metadata).is_err());
    assert!(HadamardConfig::from_metadata(&file.metadata)
        .expect("no prism.hadamard.version key: must be Ok(None), not an error")
        .is_none());
}

// ── Real dense Bonsai/Ternary-Bonsai models (M-34) ─────────────────────────

/// Compare every field `Qwen3Config::from_metadata_and_tensors` can read
/// from a real file against the corresponding named constructor's hard-coded
/// numbers, skipping `model_name` (cosmetic; not what M-34 is about) and
/// `architecture` (constructors always say `"qwen3"`; the real ternary files
/// may resolve it via the legacy `llm.*` path but land on the same string).
fn assert_named_constructor_matches_real_file(
    file_label: &str,
    real: &Qwen3Config,
    hardcoded: &Qwen3Config,
) {
    assert_eq!(
        real.hidden_size, hardcoded.hidden_size,
        "{file_label}: hidden_size"
    );
    assert_eq!(
        real.num_layers, hardcoded.num_layers,
        "{file_label}: num_layers"
    );
    assert_eq!(
        real.num_attention_heads, hardcoded.num_attention_heads,
        "{file_label}: num_attention_heads"
    );
    assert_eq!(
        real.num_kv_heads, hardcoded.num_kv_heads,
        "{file_label}: num_kv_heads"
    );
    assert_eq!(real.head_dim, hardcoded.head_dim, "{file_label}: head_dim");
    assert_eq!(
        real.value_length, hardcoded.value_length,
        "{file_label}: value_length"
    );
    assert_eq!(
        real.intermediate_size, hardcoded.intermediate_size,
        "{file_label}: intermediate_size"
    );
    assert_eq!(
        real.vocab_size, hardcoded.vocab_size,
        "{file_label}: vocab_size"
    );
    assert_eq!(
        real.max_context_length, hardcoded.max_context_length,
        "{file_label}: max_context_length"
    );
    assert_eq!(
        real.rope_scaling, hardcoded.rope_scaling,
        "{file_label}: rope_scaling"
    );
    // M-17: the named constructors all say `None`; this is the assertion
    // that keeps that honest against the shipped file, so a future model
    // that *does* declare `<arch>.attention.sliding_window` fails here
    // rather than silently running fully-causal attention.
    assert_eq!(
        real.sliding_window, hardcoded.sliding_window,
        "{file_label}: sliding_window"
    );
}

fn load_real_config(path: &Path) -> Qwen3Config {
    let mmap =
        mmap_gguf_file(path).unwrap_or_else(|e| panic!("failed to mmap {}: {e}", path.display()));
    let file = GgufFile::parse(&mmap)
        .unwrap_or_else(|e| panic!("failed to parse {}: {e}", path.display()));
    Qwen3Config::from_metadata_and_tensors(&file.metadata, &file.tensors)
        .unwrap_or_else(|e| panic!("failed to build Qwen3Config from {}: {e}", path.display()))
}

#[test]
fn bonsai_8b_named_constructor_matches_from_metadata_on_real_file() {
    let path = models_dir().join("Bonsai-8B.gguf");
    if !path.exists() {
        eprintln!("skipping bonsai_8b_named_constructor_matches_from_metadata_on_real_file: models/Bonsai-8B.gguf not present");
        return;
    }
    let real = load_real_config(&path);
    let hardcoded = Qwen3Config::bonsai_8b();
    assert_named_constructor_matches_real_file("Bonsai-8B.gguf", &real, &hardcoded);
}

#[test]
fn ternary_bonsai_8b_named_constructor_matches_from_metadata_on_real_file() {
    let path = models_dir().join("Ternary-Bonsai-8B.gguf");
    if !path.exists() {
        eprintln!("skipping ternary_bonsai_8b_named_constructor_matches_from_metadata_on_real_file: models/Ternary-Bonsai-8B.gguf not present");
        return;
    }
    let real = load_real_config(&path);
    let hardcoded = Qwen3Config::ternary_bonsai_8b();
    assert_named_constructor_matches_real_file("Ternary-Bonsai-8B.gguf", &real, &hardcoded);
}

#[test]
fn ternary_bonsai_1_7b_named_constructor_matches_from_metadata_on_real_file() {
    let path = models_dir().join("Ternary-Bonsai-1.7B.gguf");
    if !path.exists() {
        eprintln!("skipping ternary_bonsai_1_7b_named_constructor_matches_from_metadata_on_real_file: models/Ternary-Bonsai-1.7B.gguf not present");
        return;
    }
    let real = load_real_config(&path);
    let hardcoded = Qwen3Config::ternary_bonsai_1_7b();
    assert_named_constructor_matches_real_file("Ternary-Bonsai-1.7B.gguf", &real, &hardcoded);
}
