//! Shared, honest model/build description helpers used by `info`,
//! `validate`, `run`, `chat` and `build-info`.
//!
//! Centralizes two allowlists both `cmd_info` and `cmd_validate` need to
//! close the "fabricated architecture" / "false `Validation: OK`" defects
//! (cli-02, gatekeeper REQUIRED #1):
//!
//! * [`is_known_language_model_architecture`] — until B2-02 lands a shared,
//!   required-key config with its own architecture allowlist, this is this
//!   package's own interim allowlist of `general.architecture` values that
//!   name an actual language model (as opposed to e.g. a `clip` vision
//!   projector GGUF). Keep this in sync with B2-02's allowlist once it
//!   lands (recorded in this package's `deviations`).
//! * [`unsupported_tensor_types`] — mirrors the tensor types
//!   `oxibonsai_model`'s weight loader actually has a load arm for, which is
//!   narrower than `GgufTensorType::is_executable()` (a parser-level "the
//!   format is a known id" check that currently reports `true` for ids the
//!   model loader cannot yet run end-to-end, e.g. PQ2_0/PTQ1_0 ahead of
//!   B2-09 landing their loaders in wave 3). A shared
//!   `oxibonsai_model::supported_tensor_types()` would remove this
//!   duplication; recorded as a deviation since `oxibonsai-model` is not in
//!   this package's `owned_files`.
//!
//! Also carries the resolved quant-variant + kernel-tier summary line
//! `run`/`chat` print after loading the engine (cli-16), and the
//! `oxibonsai build-info` implementation (cli-19).

use std::collections::HashMap;

use oxibonsai_core::gguf::metadata::{MetadataStore, MetadataValue};
use oxibonsai_core::GgufTensorType;

// ──────────────────────────────────────────────────────────────────────────
// Architecture allowlist (cli-02 / gatekeeper REQUIRED #1)
// ──────────────────────────────────────────────────────────────────────────

/// `general.architecture` values this build recognises as an actual
/// language model, as opposed to e.g. a CLIP vision-projector GGUF.
pub(crate) const KNOWN_LANGUAGE_MODEL_ARCHITECTURES: &[&str] = &["qwen3", "qwen35"];

/// `true` if `arch` names a language-model architecture this build knows
/// how to configure and run.
pub(crate) fn is_known_language_model_architecture(arch: &str) -> bool {
    KNOWN_LANGUAGE_MODEL_ARCHITECTURES.contains(&arch)
}

// ──────────────────────────────────────────────────────────────────────────
// Tensor-type "actually loadable" allowlist (cli-02 / gatekeeper REQUIRED #1)
// ──────────────────────────────────────────────────────────────────────────

/// Tensor types `oxibonsai_model`'s weight loader has an actual load arm
/// for today. Mirrors `crates/oxibonsai-model/src/model/weight_loaders.rs`'s
/// match arms -- that file IS in this package's `owned_files` (B2-09 owns
/// it for the wave-3 kernel-dispatch/loader integration), so this list is
/// kept in sync directly rather than as an external reference — NOT the
/// same set as `GgufTensorType::is_executable()`, see the module doc.
pub(crate) const MODEL_LOADABLE_TENSOR_TYPES: &[GgufTensorType] = &[
    GgufTensorType::F32,
    GgufTensorType::F16,
    // Gatekeeper REQUIRED #4 (waves 2+2.5 review): `weight_loaders.rs`
    // (owned by this package) loads BF16 on both the flat-tensor path
    // (`dequant_any`'s BF16 arm) and the output/LM-head path
    // (`load_output_weight`'s `F32 | F16 | BF16` arm), so omitting it here
    // made `oxibonsai validate`/`info` print "no load path in this build:
    // BF16" for a type this build genuinely loads.
    GgufTensorType::BF16,
    GgufTensorType::Q1_0_g128,
    GgufTensorType::TQ2_0_g128,
    // B2-09: `weight_loaders.rs::load_transformer_block` now has real
    // `Linear{PQ2_0,PTQ1_0,Q2_0G64}` arms for these four RESOLVED types
    // (`Q2_0G128DFirst` is the PrismML gen-1 reading of ambiguous ggml id
    // 42, wire-identical to `PQ2_0`; `Q2_0G64` is the mainline group-64
    // reading of the same ambiguous id) -- omitting them here made
    // `oxibonsai validate`/`info` print "no load path in this build:
    // PQ2_0" (etc.) for a Bonsai-2 27B PQ2_0/PTQ1_0 file this build
    // genuinely loads.
    GgufTensorType::PQ2_0,
    GgufTensorType::PTQ1_0,
    GgufTensorType::Q2_0G64,
    GgufTensorType::Q2_0G128DFirst,
    GgufTensorType::Q4_0,
    GgufTensorType::Q8_0,
    GgufTensorType::Q2_K,
    GgufTensorType::Q3_K,
    GgufTensorType::Q4_K,
    GgufTensorType::Q5_K,
    GgufTensorType::Q6_K,
    GgufTensorType::Q8_K,
    GgufTensorType::F8_E4M3,
    GgufTensorType::F8_E5M2,
];

/// Return every distinct [`GgufTensorType`] present in `type_counts` that
/// the model loader cannot execute today, sorted (by display form) for
/// stable, deterministic output.
pub(crate) fn unsupported_tensor_types(
    type_counts: &HashMap<GgufTensorType, usize>,
) -> Vec<GgufTensorType> {
    let mut out: Vec<GgufTensorType> = type_counts
        .keys()
        .filter(|ty| !MODEL_LOADABLE_TENSOR_TYPES.contains(ty))
        .copied()
        .collect();
    out.sort_by_key(|ty| ty.to_string());
    out
}

// ──────────────────────────────────────────────────────────────────────────
// Honest metadata display ("print only values actually present")
// ──────────────────────────────────────────────────────────────────────────

/// Render a single [`MetadataValue`] for human display.
///
/// Arrays longer than 8 elements are summarised as `[N items]` rather than
/// dumped in full: this is a human debug display for `info`/`validate`, not
/// a machine format, and a multi-thousand-entry tokenizer vocab/merges
/// array would drown everything else on the screen.
pub(crate) fn format_metadata_value(value: &MetadataValue) -> String {
    match value {
        MetadataValue::Uint8(v) => v.to_string(),
        MetadataValue::Int8(v) => v.to_string(),
        MetadataValue::Uint16(v) => v.to_string(),
        MetadataValue::Int16(v) => v.to_string(),
        MetadataValue::Uint32(v) => v.to_string(),
        MetadataValue::Int32(v) => v.to_string(),
        MetadataValue::Uint64(v) => v.to_string(),
        MetadataValue::Int64(v) => v.to_string(),
        MetadataValue::Float32(v) => v.to_string(),
        MetadataValue::Float64(v) => v.to_string(),
        MetadataValue::Bool(v) => v.to_string(),
        MetadataValue::String(v) => v.clone(),
        MetadataValue::Array(items) => {
            if items.len() <= 8 {
                let rendered: Vec<String> = items.iter().map(format_metadata_value).collect();
                format!("[{}]", rendered.join(", "))
            } else {
                format!("[{} items]", items.len())
            }
        }
    }
}

/// Look up `key` in `metadata` and format it for display, or `"-"` when the
/// key is absent — the honest "print only values actually present"
/// contract cli-02 asks for, instead of a `Qwen3Config`-style default
/// silently standing in for a value that was never in the file.
pub(crate) fn display_metadata_value(metadata: &MetadataStore, key: &str) -> String {
    metadata
        .get(key)
        .map(format_metadata_value)
        .unwrap_or_else(|| "-".to_string())
}

/// One architecture-scoped numeric field, probed the same way
/// [`oxibonsai_core::config::Qwen3Config::from_metadata`] resolves it
/// (`"{arch}.{arch_suffix}"`, falling back to the generic `"llm.{suffix}"`
/// key) — except this returns the honest display string `"-"` instead of
/// substituting a hardcoded default when neither key is present.
pub(crate) fn display_arch_scoped_u32(
    metadata: &MetadataStore,
    arch: &str,
    arch_suffix: &str,
    generic_key: &str,
) -> String {
    metadata
        .get_u32(&format!("{arch}.{arch_suffix}"))
        .ok()
        .or_else(|| metadata.get_u32(generic_key).ok())
        .map(|v| v.to_string())
        .unwrap_or_else(|| "-".to_string())
}

// ──────────────────────────────────────────────────────────────────────────
// Resolved engine summary (cli-16: report the real quant variant + kernel
// tier, never a hardcoded kernel-family string)
// ──────────────────────────────────────────────────────────────────────────

/// Build the one-line "what actually loaded" summary `run`/`chat`/`benchmark`
/// print right after constructing the engine: the RESOLVED dominant quant
/// variant (derived from the model's own tensor types), the exact
/// [`oxibonsai_core::GgufTensorType`] that variant was resolved from, and
/// the effective [`oxibonsai_kernels::KernelTier`] with the dispatcher's own
/// reason.
///
/// The quant type is included (not just the variant name) because
/// `ModelVariant::name()` alone can print the uninformative `"Custom"` for
/// any tensor layout the classifier does not recognize as one of its named
/// presets — verified live on a real model, that used to render as
/// `"Resolved model: Custom | kernel tier: neon (...)"`, telling an
/// operator nothing about what actually loaded (cli-16).
///
/// Printed by the CLI itself rather than relying solely on the runtime's
/// `"inference engine loaded from GGUF kernel=..."` log line
/// (`oxibonsai-runtime/src/engine.rs`, not in this package's `owned_files`),
/// which still hardcodes the 1-bit family name in that label — see this
/// package's `deviations`.
pub(crate) fn resolved_engine_summary(
    variant_name: &str,
    dominant_quant_type: oxibonsai_core::GgufTensorType,
    kernel_tier: oxibonsai_kernels::KernelTier,
    kernel_tier_reason: &str,
) -> String {
    format!(
        "Resolved model: {variant_name} ({dominant_quant_type}) | kernel tier: {kernel_tier} \
         ({kernel_tier_reason})"
    )
}

// ──────────────────────────────────────────────────────────────────────────
// `oxibonsai build-info` (cli-19)
// ──────────────────────────────────────────────────────────────────────────

/// `oxibonsai build-info` — print what this binary was built with: enabled
/// Cargo features, which kernel tiers are compiled in, the tier this
/// process actually detects at runtime (with the dispatcher's own reason,
/// so perf-13's silent CPU-only degradation becomes visible instead of
/// silent), and a best-effort git commit hash.
///
/// Deliberately a standalone subcommand rather than `info --build`: `info`
/// requires `--model`/`OXI_MODEL`, but build information should be
/// obtainable with no model present at all.
pub(crate) fn print_build_info() {
    println!("oxibonsai {}", env!("CARGO_PKG_VERSION"));
    println!();

    println!("Build features:");
    for (name, enabled) in build_features() {
        println!("  {name}: {}", if enabled { "on" } else { "off" });
    }
    println!();

    println!(
        "Compiled-in kernel tiers: {}",
        compiled_kernel_tiers().join(", ")
    );

    let dispatcher = oxibonsai_kernels::KernelDispatcher::auto_detect();
    println!(
        "Detected runtime tier: {} ({})",
        dispatcher.tier(),
        dispatcher.effective_tier_reason()
    );

    // cli-08: makes the CLI's own `native-tokenizer` feature's effect
    // observable — which backend `--tokenizer-backend auto` (the default)
    // resolves to in this exact build.
    println!(
        "Default tokenizer backend (--tokenizer-backend auto): {}",
        super::tokenizer_backend::active_backend_name(
            super::tokenizer_backend::TokenizerBackendChoice::Auto
        )
    );
    println!();

    println!("Git commit: {}", git_commit_best_effort());
}

/// This binary's own Cargo feature flags and whether each is compiled in.
fn build_features() -> Vec<(&'static str, bool)> {
    vec![
        ("server", cfg!(feature = "server")),
        ("rag", cfg!(feature = "rag")),
        ("eval", cfg!(feature = "eval")),
        ("hf-tokenizer", cfg!(feature = "hf-tokenizer")),
        ("native-tokenizer", cfg!(feature = "native-tokenizer")),
        ("metal", cfg!(feature = "metal")),
        ("native-cuda", cfg!(feature = "native-cuda")),
        ("cuda", cfg!(feature = "cuda")),
        ("gpu", cfg!(feature = "gpu")),
        ("simd-avx2", cfg!(feature = "simd-avx2")),
        ("simd-avx512", cfg!(feature = "simd-avx512")),
        ("simd-neon", cfg!(feature = "simd-neon")),
        ("wasm", cfg!(feature = "wasm")),
    ]
}

/// Kernel tiers this specific binary was compiled with support for
/// (independent of which one the CPU it happens to run on actually uses —
/// see [`print_build_info`]'s "Detected runtime tier" line for that).
fn compiled_kernel_tiers() -> Vec<&'static str> {
    let mut tiers = vec!["reference"];
    if cfg!(all(target_arch = "x86_64", feature = "simd-avx2")) {
        tiers.push("avx2+fma");
    }
    if cfg!(all(target_arch = "x86_64", feature = "simd-avx512")) {
        tiers.push("avx512f+bw+vl");
    }
    if cfg!(all(target_arch = "aarch64", feature = "simd-neon")) {
        tiers.push("neon");
    }
    if cfg!(feature = "gpu") {
        tiers.push("gpu");
    }
    tiers
}

/// Best-effort git short commit hash, shelled out to `git rev-parse` at
/// RUNTIME rather than baked in by a `build.rs`.
///
/// A `build.rs` would report the commit the binary was *compiled* from
/// (the more correct answer) but is a new file outside this package's
/// `owned_files` this wave; recorded in `deviations`. This runtime
/// fallback instead reports the checkout the binary happens to be *run*
/// from — the right answer for `cargo run`/a workspace binary invoked from
/// the repo root, but not necessarily meaningful after `cargo install`
/// (which ships no `.git` at all) or when run from an unrelated directory.
/// Both of those report "unknown" honestly rather than a wrong guess.
fn git_commit_best_effort() -> String {
    match std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
    {
        Ok(output) if output.status.success() && !output.stdout.is_empty() => {
            String::from_utf8_lossy(&output.stdout).trim().to_string()
        }
        _ => "unknown (not run from a git checkout, or git is not installed)".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_architectures_accepts_qwen_variants() {
        assert!(is_known_language_model_architecture("qwen3"));
        assert!(is_known_language_model_architecture("qwen35"));
    }

    #[test]
    fn known_architectures_rejects_non_language_model_archs() {
        assert!(!is_known_language_model_architecture("clip"));
        assert!(!is_known_language_model_architecture(""));
        assert!(!is_known_language_model_architecture("llama"));
    }

    /// B2-09 landed real `weight_loaders.rs` loaders for PTQ1_0/PQ2_0/
    /// Q2_0G64/Q2_0G128DFirst (see
    /// `prism_ternary_family_is_not_flagged_unsupported` below), so this
    /// test -- which used to assert PTQ1_0 was flagged unsupported, true
    /// only *before* those loaders landed -- now uses mainline `TQ2_0`
    /// (ggml id 35) as its still-genuinely-unsupported example:
    /// `weight_loaders.rs` can `dequant_any` it but has no
    /// transformer-block/output-projection `Linear*` wrapper for it in any
    /// build.
    #[test]
    fn unsupported_tensor_types_flags_types_with_no_load_arm() {
        let mut counts = HashMap::new();
        counts.insert(GgufTensorType::TQ2_0_g128, 100usize);
        counts.insert(GgufTensorType::TQ2_0, 5usize);
        let unsupported = unsupported_tensor_types(&counts);
        assert_eq!(unsupported, vec![GgufTensorType::TQ2_0]);
    }

    #[test]
    fn unsupported_tensor_types_empty_when_all_loadable() {
        let mut counts = HashMap::new();
        counts.insert(GgufTensorType::F32, 1usize);
        counts.insert(GgufTensorType::TQ2_0_g128, 10usize);
        assert!(unsupported_tensor_types(&counts).is_empty());
    }

    /// Gatekeeper REQUIRED #4: BF16 is loaded by `weight_loaders.rs`'s
    /// `dequant_any`/`load_output_weight` (`ssm_alpha`/`ssm_beta` flat
    /// tensors and an FP32-widened output/LM-head), so it must not be
    /// reported as unsupported.
    #[test]
    fn bf16_is_not_flagged_unsupported() {
        let mut counts = HashMap::new();
        counts.insert(GgufTensorType::BF16, 96usize);
        assert!(unsupported_tensor_types(&counts).is_empty());
        assert!(MODEL_LOADABLE_TENSOR_TYPES.contains(&GgufTensorType::BF16));
    }

    /// B2-09 blocking fix: `weight_loaders.rs::load_transformer_block` gained
    /// real arms for the four RESOLVED ambiguous-id-42 / PrismML PQ2_0
    /// family types this wave, so none of them may be reported as
    /// unsupported (a Bonsai-2 27B PQ2_0/PTQ1_0 file must not print "no
    /// load path in this build: PQ2_0").
    #[test]
    fn prism_ternary_family_is_not_flagged_unsupported() {
        let mut counts = HashMap::new();
        counts.insert(GgufTensorType::PQ2_0, 401usize);
        counts.insert(GgufTensorType::PTQ1_0, 401usize);
        counts.insert(GgufTensorType::Q2_0G64, 401usize);
        counts.insert(GgufTensorType::Q2_0G128DFirst, 401usize);
        assert!(unsupported_tensor_types(&counts).is_empty());
        for ty in [
            GgufTensorType::PQ2_0,
            GgufTensorType::PTQ1_0,
            GgufTensorType::Q2_0G64,
            GgufTensorType::Q2_0G128DFirst,
        ] {
            assert!(
                MODEL_LOADABLE_TENSOR_TYPES.contains(&ty),
                "{ty} must be in MODEL_LOADABLE_TENSOR_TYPES"
            );
        }
    }

    #[test]
    fn format_metadata_value_renders_scalars() {
        assert_eq!(format_metadata_value(&MetadataValue::Uint32(42)), "42");
        assert_eq!(
            format_metadata_value(&MetadataValue::String("qwen3".to_string())),
            "qwen3"
        );
        assert_eq!(format_metadata_value(&MetadataValue::Bool(true)), "true");
    }

    #[test]
    fn format_metadata_value_summarises_large_arrays() {
        let items: Vec<MetadataValue> = (0..100).map(MetadataValue::Uint32).collect();
        let rendered = format_metadata_value(&MetadataValue::Array(items));
        assert_eq!(rendered, "[100 items]");
    }

    #[test]
    fn format_metadata_value_renders_small_arrays_in_full() {
        let items = vec![MetadataValue::Uint32(1), MetadataValue::Uint32(2)];
        let rendered = format_metadata_value(&MetadataValue::Array(items));
        assert_eq!(rendered, "[1, 2]");
    }

    #[test]
    fn resolved_engine_summary_mentions_variant_and_tier() {
        let summary = resolved_engine_summary(
            "Ternary-Bonsai-1.7B",
            oxibonsai_core::GgufTensorType::Q1_0_g128,
            oxibonsai_kernels::KernelTier::Reference,
            "no SIMD feature compiled in",
        );
        assert!(summary.contains("Ternary-Bonsai-1.7B"));
        assert!(summary.contains("reference"));
        assert!(summary.contains("no SIMD feature compiled in"));
    }

    #[test]
    fn resolved_engine_summary_names_the_resolved_quant_type() {
        // cli-16: the summary used to print only the `ModelVariant` name,
        // which renders as the uninformative "Custom" for any tensor
        // layout the classifier does not special-case — the quant type
        // actually resolved from the file's own tensors must be visible
        // even then.
        let summary = resolved_engine_summary(
            "Custom",
            oxibonsai_core::GgufTensorType::Q1_0_g128,
            oxibonsai_kernels::KernelTier::Neon,
            "aarch64 NEON detected",
        );
        assert!(
            summary.contains("Q1_0_g128"),
            "summary must name the resolved quant type; got: {summary}"
        );
    }

    #[test]
    fn build_info_does_not_panic() {
        // Smoke test: build-info touches process/env/CPU-feature detection
        // and must never panic regardless of the environment it runs in.
        print_build_info();
    }
}
