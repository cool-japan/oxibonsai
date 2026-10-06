//! Shared, honest model/build description helpers used by `info`,
//! `validate`, `run`, `chat` and `build-info`.
//!
//! Centralizes two allowlists both `cmd_info` and `cmd_validate` need to
//! close the "fabricated architecture" / "false `Validation: OK`" defects
//! (cli-02):
//!
//! * [`is_known_language_model_architecture`] — an interim allowlist of
//!   `general.architecture` values that name an actual language model (as
//!   opposed to e.g. a `clip` vision projector GGUF), pending a shared,
//!   required-key config with its own architecture allowlist elsewhere in
//!   the workspace.
//! * [`unsupported_tensor_types`] — mirrors the tensor types
//!   `oxibonsai_model`'s weight loader actually has a load arm for, which is
//!   narrower than `GgufTensorType::is_executable()` (a parser-level "the
//!   format is a known id" check that currently reports `true` for ids the
//!   model loader cannot yet run end-to-end, e.g. PQ2_0/PTQ1_0 before their
//!   loaders landed). A shared `oxibonsai_model::supported_tensor_types()`
//!   would remove this duplication.
//!
//! Also carries the resolved quant-variant + kernel-tier summary line
//! `run`/`chat` print after loading the engine (cli-16), and the
//! `oxibonsai build-info` implementation (cli-19).

use std::collections::HashMap;

use oxibonsai_core::gguf::metadata::{MetadataStore, MetadataValue};
use oxibonsai_core::GgufTensorType;

// ──────────────────────────────────────────────────────────────────────────
// Architecture allowlist (cli-02)
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
// Tensor-type "actually loadable" allowlist (cli-02)
// ──────────────────────────────────────────────────────────────────────────

/// Tensor types `oxibonsai_model`'s weight loader has an actual load arm
/// for today. Mirrors `crates/oxibonsai-model/src/model/weight_loaders.rs`'s
/// match arms, kept in sync by hand rather than as an external reference
/// since that file lives in a different crate — NOT the same set as
/// `GgufTensorType::is_executable()`, see the module doc.
pub(crate) const MODEL_LOADABLE_TENSOR_TYPES: &[GgufTensorType] = &[
    GgufTensorType::F32,
    GgufTensorType::F16,
    // `weight_loaders.rs` loads BF16 on both the flat-tensor path
    // (`dequant_any`'s BF16 arm) and the output/LM-head path
    // (`load_output_weight`'s `F32 | F16 | BF16` arm), so omitting it here
    // made `oxibonsai validate`/`info` print "no load path in this build:
    // BF16" for a type this build genuinely loads.
    GgufTensorType::BF16,
    GgufTensorType::Q1_0_g128,
    GgufTensorType::TQ2_0_g128,
    // `weight_loaders.rs::load_transformer_block` has real
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
/// (`oxibonsai-runtime/src/engine.rs`), which still hardcodes the 1-bit
/// family name in that label.
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

/// The summary line `run`/`chat`/`benchmark` print and `serve` logs after
/// building an engine (cli-16), from the ENGINE's own
/// accessors: the resolved variant (the hybrid model's own detection, or
/// `BonsaiModel::variant()` — never the raw parse-time tensor type, which
/// named the 27B "Custom"), the resolved dominant quant type, the effective
/// kernel tier with its reason, the kernel label and the model description.
pub(crate) fn engine_summary(engine: &oxibonsai_runtime::InferenceEngine<'_>) -> String {
    let variant = match (engine.hybrid_model(), engine.dense_model()) {
        (Some(hybrid), _) => hybrid.variant().map(|v| v.name().to_string()),
        (None, Some(dense)) => Some(dense.variant().name().to_string()),
        (None, None) => None,
    }
    .unwrap_or_else(|| engine.architecture().to_string());
    format!(
        "{} | kernel: {} | {}",
        resolved_engine_summary(
            &variant,
            engine.dominant_quant_type(),
            engine.kernel_tier(),
            &engine.effective_tier_reason(),
        ),
        engine.kernel_label(),
        engine.model_description(),
    )
}

// ──────────────────────────────────────────────────────────────────────────
// Truthful `qwen35` hybrid report
// ──────────────────────────────────────────────────────────────────────────

/// The `prism.hadamard.*` contract a Bonsai 2 file declares.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HadamardSummary {
    /// `prism.hadamard.version`.
    pub(crate) version: u32,
    /// `prism.hadamard.block_size` (1024).
    pub(crate) block_size: usize,
    /// The rotated input widths with a sign vector (5120, 6144, 17408).
    pub(crate) sign_widths: Vec<usize>,
    /// Folded weight matrices (`weight_names`, 401 for the 27B).
    pub(crate) folded: usize,
    /// Inverse-rotated tensors (`inverse_weight_names`: `token_embd.weight`).
    pub(crate) inverse: usize,
    /// `prism.hadamard.gdn_v_grouped`.
    pub(crate) gdn_v_grouped: bool,
}

/// Everything `info`/`validate` report about a `qwen35` hybrid.
#[derive(Debug, Clone)]
pub(crate) struct HybridReport {
    pub(crate) layers: usize,
    pub(crate) full_layers: Vec<usize>,
    pub(crate) linear_layers: usize,
    pub(crate) vocab: usize,
    pub(crate) context_length: usize,
    pub(crate) hadamard: Option<HadamardSummary>,
    pub(crate) geometry: super::bonsai2::HybridStateGeometry,
    /// The dry bind (`HybridModel::from_gguf`): the resolved quant type,
    /// variant and folded count — or why the model cannot be bound.
    pub(crate) bind: Result<HybridBind, String>,
    /// The CPU tier the CPU model runs on (`--backend cpu`, the embedding
    /// pass, and `--backend auto` when no Metal runner serves the model).
    pub(crate) kernel_tier: oxibonsai_kernels::KernelTier,
    /// What `--backend auto` does with the model on this host at the
    /// default `--ctx`: the Metal hybrid runner with its KV window, or the
    /// CPU tier and why (`None` when the model did not bind).
    pub(crate) backend_plan: Option<oxibonsai_runtime::engine_hybrid_gpu::HybridBackendPlan>,
    /// `--prefill-chunk` as the report was asked for (`None`: the model's
    /// own default).
    pub(crate) prefill_chunk_requested: Option<usize>,
    /// The model's own prefill chunk (`None` when the model did not bind).
    pub(crate) default_prefill_chunk: Option<usize>,
}

/// What a successful dry bind resolved.
#[derive(Debug, Clone)]
pub(crate) struct HybridBind {
    pub(crate) quant: GgufTensorType,
    pub(crate) variant: Option<String>,
    pub(crate) folded: usize,
    pub(crate) description: String,
}

/// KV cache length the dry bind allocates (tiny: the bind validates the
/// weights, it never decodes).
const DRY_BIND_CONTEXT: usize = 16;

/// Build the truthful hybrid report for a parsed `qwen35` GGUF: the config
/// and Hadamard contract from metadata, plus a header-only dry bind of the
/// hybrid model — the constructor `run` actually uses (weights stay in the
/// memory map; ~1-2 s for the 27B) — and the backend `--backend auto`
/// resolves to for it at the default `--ctx` (the Metal hybrid runner's
/// footprint and window, computed without building one; the Metal device is
/// opened only to read its limits), under the `HybridLoadOptions` in force
/// on this thread — a vision tower's resident bytes and the prefill chunk
/// the runner's calls are sized for.
///
/// # Errors
///
/// The file's `qwen35` hyper-parameters do not parse (a malformed Hadamard
/// contract or an unbindable model is reported in the result instead).
pub(crate) fn hybrid_report(
    gguf: &oxibonsai_core::gguf::reader::GgufFile<'_>,
) -> anyhow::Result<HybridReport> {
    let cfg = oxibonsai_core::config_hybrid::HybridConfig::from_metadata(&gguf.metadata)
        .map_err(|e| anyhow::anyhow!("qwen35 hyper-parameters: {e}"))?;
    let geometry = super::bonsai2::HybridStateGeometry::from_config(&cfg);
    let full_layers: Vec<usize> = (0..cfg.base.num_layers)
        .filter(|&layer| cfg.is_full_attention(layer))
        .collect();
    let hadamard =
        match oxibonsai_core::hadamard_config::HadamardConfig::from_metadata(&gguf.metadata) {
            Ok(Some(h)) => {
                let mut sign_widths: Vec<usize> = h.signs.keys().copied().collect();
                sign_widths.sort_unstable();
                Some(HadamardSummary {
                    version: gguf.metadata.get_u32("prism.hadamard.version").unwrap_or(0),
                    block_size: h.block_size,
                    sign_widths,
                    folded: h.folded.len(),
                    inverse: h.inverse.len(),
                    gdn_v_grouped: h.gdn_v_grouped,
                })
            }
            Ok(None) => None,
            Err(e) => anyhow::bail!("prism.hadamard.* contract: {e}"),
        };
    let default_ctx = super::bonsai2::default_max_seq_len("qwen35");
    let (bind, backend_plan, default_prefill_chunk) =
        match oxibonsai_model::hybrid::HybridModel::from_gguf(gguf, DRY_BIND_CONTEXT) {
            Ok(model) => (
                Ok(HybridBind {
                    quant: model.quant_type(),
                    variant: model.variant().map(|v| v.name().to_string()),
                    folded: model.folded_count(),
                    description: model.describe(),
                }),
                Some(oxibonsai_runtime::engine_hybrid_gpu::hybrid_backend_plan(
                    gguf,
                    &model,
                    default_ctx,
                )),
                Some(model.prefill_chunk()),
            ),
            Err(e) => (Err(e.to_string()), None, None),
        };
    Ok(HybridReport {
        layers: cfg.base.num_layers,
        linear_layers: cfg.num_linear_layers(),
        full_layers,
        vocab: cfg.base.vocab_size,
        context_length: cfg.base.max_context_length,
        hadamard,
        geometry,
        bind,
        kernel_tier: oxibonsai_kernels::cpu_kernel_tier(),
        backend_plan,
        prefill_chunk_requested: oxibonsai_runtime::engine_hybrid_gpu::HybridLoadScope::active()
            .prefill_chunk,
        default_prefill_chunk,
    })
}

/// The prefill chunk a hybrid report's engine would run in: the tokens one
/// prefill call takes (`in_effect`), the request it came from (`None`: the
/// model's default), and whether the executor capped the request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PrefillChunkPlan {
    /// `--prefill-chunk`, when one was asked for.
    pub(crate) requested: Option<usize>,
    /// The model's chunk: the request, else the model's own default.
    pub(crate) model_chunk: usize,
    /// The most tokens one prefill call takes on the planned executor.
    pub(crate) in_effect: usize,
}

impl PrefillChunkPlan {
    /// Whether the planned executor takes fewer tokens per call than the
    /// model's chunk (the Metal runner's KV-window budget capped it).
    pub(crate) fn capped(&self) -> bool {
        self.in_effect < self.model_chunk
    }

    /// The report line.
    pub(crate) fn describe(&self) -> String {
        let source = match self.requested {
            Some(requested) => format!("--prefill-chunk {requested}"),
            None => "the model's default; --prefill-chunk changes it".to_string(),
        };
        if self.capped() {
            format!(
                "{} tokens per prefill call ({source}): larger calls would not leave room for \
                 the KV window in the memory budget",
                self.in_effect
            )
        } else {
            format!("{} tokens per prefill call ({source})", self.in_effect)
        }
    }
}

/// The effective decode tier of a hybrid under `--backend auto`, and the
/// one-line reason `info` prints for it (the reason always names the model
/// kind, "hybrid qwen35 model").
pub(crate) fn hybrid_auto_tier(report: &HybridReport) -> (String, String) {
    use oxibonsai_runtime::engine_hybrid_gpu::{HybridBackendPlan, HYBRID_RUNNER_LABEL};
    match &report.backend_plan {
        Some(HybridBackendPlan::Metal { window, .. }) => (
            "gpu".to_string(),
            format!(
                "hybrid qwen35 model: executor {HYBRID_RUNNER_LABEL} under --backend auto/metal \
                 (KV window {} positions at the default --ctx, device ceiling {} positions for a \
                 runner alone); --backend cpu runs it on the {} CPU tier",
                window.window, window.device_ceiling, report.kernel_tier
            ),
        ),
        Some(HybridBackendPlan::Cpu { reason }) => (
            report.kernel_tier.to_string(),
            format!(
                "hybrid qwen35 model: runs on the best CPU tier under --backend auto/cpu, and \
                 --backend metal is refused, because the Metal hybrid runner does not serve it \
                 here: {reason}"
            ),
        ),
        None => (
            report.kernel_tier.to_string(),
            "hybrid qwen35 model: the dry bind failed, so no backend was resolved".to_string(),
        ),
    }
}

impl HybridReport {
    /// The prefill chunk the planned executor would run in (`None` when the
    /// model did not bind): on the Metal runner its call size, sized for the
    /// request within the KV window's memory budget; on the CPU model the
    /// model's chunk.
    pub(crate) fn prefill_chunk_plan(&self) -> Option<PrefillChunkPlan> {
        use oxibonsai_runtime::engine_hybrid_gpu::HybridBackendPlan;
        let model_chunk = self
            .prefill_chunk_requested
            .or(self.default_prefill_chunk)?;
        let in_effect = match &self.backend_plan {
            Some(HybridBackendPlan::Metal { call_tokens, .. }) => *call_tokens,
            Some(HybridBackendPlan::Cpu { .. }) => model_chunk,
            None => return None,
        };
        Some(PrefillChunkPlan {
            requested: self.prefill_chunk_requested,
            model_chunk,
            in_effect,
        })
    }

    /// Human-readable report lines (shared by `info` and `validate`).
    pub(crate) fn lines(&self, weight_bytes: u64) -> Vec<String> {
        use super::bonsai2::gib;
        let mut out = Vec::new();
        out.push(format!(
            "Hybrid layers: {} ({} full attention / {} Gated-DeltaNet); full-attention layers {:?}",
            self.layers,
            self.full_layers.len(),
            self.linear_layers,
            self.full_layers
        ));
        match &self.bind {
            Ok(bind) => {
                out.push(format!(
                    "Weights: {} (ggml type id {}){}; {} folded tensors bound",
                    bind.quant,
                    bind.quant.wire_id(),
                    bind.variant
                        .as_deref()
                        .map(|v| format!(", variant {v}"))
                        .unwrap_or_default(),
                    bind.folded
                ));
                out.push(format!("Model: {}", bind.description));
            }
            Err(e) => out.push(format!("Hybrid bind: FAILED ({e})")),
        }
        match &self.hadamard {
            Some(h) => out.push(format!(
                "Hadamard: prism.hadamard.version {} | block_size {} | sign_widths {:?} | {} \
                 weight_names | {} inverse_weight_names | gdn_v_grouped {}",
                h.version, h.block_size, h.sign_widths, h.folded, h.inverse, h.gdn_v_grouped
            )),
            None => out.push("Hadamard: none (no prism.hadamard.* contract)".to_string()),
        }
        out.push(format!(
            "Vocab: {} | declared context: {} tokens",
            self.vocab, self.context_length
        ));
        let default_ctx = super::bonsai2::default_max_seq_len("qwen35");
        let ram = oxibonsai_runtime::config::total_ram_bytes();
        let ram_limit = ram.map(|total| {
            oxibonsai_runtime::config::max_context_for_budget(
                total,
                weight_bytes,
                self.geometry.recurrent_bytes,
                self.geometry.kv_bytes_per_token,
                self.context_length,
            )
        });
        out.push(format!(
            "Per-sequence state: KV {} bytes/token (f16) = {} at the default --ctx {default_ctx}, \
             {} at the declared {}; recurrent {} bytes ({}); RAM-derived max --ctx on this host: {}",
            self.geometry.kv_bytes_per_token,
            gib(self.geometry.kv_bytes_at(default_ctx)),
            gib(self.geometry.kv_bytes_at(self.context_length)),
            self.context_length,
            self.geometry.recurrent_bytes,
            gib(self.geometry.recurrent_bytes),
            ram_limit.map_or_else(|| "unknown".to_string(), |l| l.to_string()),
        ));
        let (tier, reason) = hybrid_auto_tier(self);
        out.push(format!("Kernel tier: {tier} ({reason})"));
        if let Some(plan) = self.prefill_chunk_plan() {
            out.push(format!("Prefill chunk: {}", plan.describe()));
        }
        if let Some(oxibonsai_runtime::engine_hybrid_gpu::HybridBackendPlan::Metal {
            window,
            mapped,
            ..
        }) = &self.backend_plan
        {
            out.push(format!(
                "Backend: {} — {}",
                oxibonsai_runtime::engine_hybrid_gpu::HYBRID_RUNNER_LABEL,
                window.summary()
            ));
            out.push(format!(
                "Metal residents: {}; the runner reads the weights {} and allocates its f16 KV \
                 for the whole window at load ({} bytes/position: {} for the {}-position window \
                 the default --ctx {} wires)",
                window.residents.describe(),
                if *mapped {
                    "in place from the file mapping"
                } else {
                    "from a copy (the image is not page-aligned)"
                },
                if window.window == 0 {
                    0
                } else {
                    window.runner_kv_bytes / window.window as u64
                },
                gib(window.runner_kv_bytes),
                window.window,
                window.requested,
            ));
        }
        out
    }

    /// Machine-readable form for `info --json`.
    pub(crate) fn to_json(&self, weight_bytes: u64) -> serde_json::Value {
        let ram_limit = oxibonsai_runtime::config::total_ram_bytes().map(|total| {
            oxibonsai_runtime::config::max_context_for_budget(
                total,
                weight_bytes,
                self.geometry.recurrent_bytes,
                self.geometry.kv_bytes_per_token,
                self.context_length,
            )
        });
        let default_ctx = super::bonsai2::default_max_seq_len("qwen35");
        serde_json::json!({
            "layers": self.layers,
            "full_attention_layers": self.full_layers,
            "gated_deltanet_layers": self.linear_layers,
            "vocab_size": self.vocab,
            "context_length": self.context_length,
            "weights": match &self.bind {
                Ok(bind) => serde_json::json!({
                    "quant": bind.quant.to_string(),
                    "ggml_type_id": bind.quant.wire_id(),
                    "variant": bind.variant,
                    "folded_tensors": bind.folded,
                    "description": bind.description,
                }),
                Err(e) => serde_json::json!({ "bind_error": e }),
            },
            "hadamard": self.hadamard.as_ref().map(|h| serde_json::json!({
                "version": h.version,
                "block_size": h.block_size,
                "sign_widths": h.sign_widths,
                "weight_names": h.folded,
                "inverse_weight_names": h.inverse,
                "gdn_v_grouped": h.gdn_v_grouped,
            })),
            "kv_bytes_per_token": self.geometry.kv_bytes_per_token,
            "kv_bytes_at_default_ctx": self.geometry.kv_bytes_at(default_ctx),
            "default_ctx": default_ctx,
            "recurrent_bytes": self.geometry.recurrent_bytes,
            "ram_derived_max_ctx": ram_limit,
            "kernel_tier": hybrid_auto_tier(self).0,
            "cpu_kernel_tier": self.kernel_tier.to_string(),
            "backend": self.backend_json(),
            "prefill_chunk": self.prefill_chunk_plan().map(|plan| serde_json::json!({
                "requested": plan.requested,
                "model_chunk": plan.model_chunk,
                "in_effect": plan.in_effect,
                "capped": plan.capped(),
            })),
        })
    }

    /// The `--backend auto` resolution for `info --json`.
    fn backend_json(&self) -> serde_json::Value {
        use oxibonsai_runtime::engine_hybrid_gpu::HybridBackendPlan;
        match &self.backend_plan {
            Some(HybridBackendPlan::Metal {
                window,
                mapped,
                call_tokens,
            }) => serde_json::json!({
                "executor": "metal",
                "label": oxibonsai_runtime::engine_hybrid_gpu::HYBRID_RUNNER_LABEL,
                "weights_mapped": mapped,
                "call_tokens": call_tokens,
                "vision_resident_bytes": window.vision_resident_bytes,
                "window": window.window,
                "requested": window.requested,
                "declared": window.declared,
                "ram_guard": window.ram_guard,
                "resident_budget": window.resident_budget,
                "runner_alone_budget": window.runner_alone_budget,
                "device_ceiling": window.device_ceiling,
                "residents": window.residents.describe(),
                "limits_applied": window
                    .limits_applied
                    .iter()
                    .map(|l| l.as_str())
                    .collect::<Vec<_>>(),
                "runner_allocated_bytes": window.runner_allocated_bytes,
                "runner_kv_bytes": window.runner_kv_bytes,
            }),
            Some(HybridBackendPlan::Cpu { reason }) => serde_json::json!({
                "executor": "cpu",
                "reason": reason,
            }),
            None => serde_json::Value::Null,
        }
    }
}

/// The kernel tier `run` would use for a DENSE model under `--backend
/// auto`, with the dispatcher's own reason.
pub(crate) fn dense_auto_tier() -> (String, String) {
    let dispatcher = oxibonsai_kernels::KernelDispatcher::auto_detect();
    (
        dispatcher.tier().to_string(),
        dispatcher.effective_tier_reason(),
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
///
/// `simd-neon`/`simd-avx2`/`simd-avx512` are empty Cargo features in
/// `crates/oxibonsai-kernels/Cargo.toml` — zero `cfg(feature =
/// "simd-...")` anywhere in that crate gates on them — so a build-info line
/// keyed on those features would print `simd-neon: off` on a real M3 even
/// though `KernelDispatcher::auto_detect()` picks NEON there every time.
/// The real selection is `target_arch`-based
/// (`oxibonsai-kernels/src/dispatch.rs`'s `cpu_tier()`: NEON is
/// unconditionally available on `aarch64`, and AVX2/AVX512 are attempted
/// unconditionally on `x86_64` via runtime `is_x86_feature_detected!`, with
/// no Cargo feature gate on either target) — so this mirrors that same
/// `target_arch` predicate instead of the dead features.
fn compiled_kernel_tiers() -> Vec<&'static str> {
    let mut tiers = vec!["reference"];
    if cfg!(target_arch = "x86_64") {
        tiers.push("avx2+fma");
        tiers.push("avx512f+bw+vl");
    }
    if cfg!(target_arch = "aarch64") {
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
/// (the more correct answer), but that is a larger change than this runtime
/// fallback, which instead reports the checkout the binary happens to be *run*
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

    /// `weight_loaders.rs` has real loaders for PTQ1_0/PQ2_0/Q2_0G64/
    /// Q2_0G128DFirst (see `prism_ternary_family_is_not_flagged_unsupported`
    /// below), so this test uses mainline `TQ2_0` (ggml id 35) as its
    /// still-genuinely-unsupported example:
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

    /// BF16 is loaded by `weight_loaders.rs`'s
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

    /// `weight_loaders.rs::load_transformer_block` has real arms for the
    /// four RESOLVED ambiguous-id-42 / PrismML PQ2_0 family types, so none
    /// of them may be reported as unsupported (a Bonsai-2 27B PQ2_0/PTQ1_0
    /// file must not print "no load path in this build: PQ2_0").
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
            oxibonsai_kernels::KernelTier::Reference,
            "no SIMD feature compiled in",
        );
        assert!(
            summary.contains("Q1_0_g128"),
            "summary must name the resolved quant type; got: {summary}"
        );
    }

    /// `compiled_kernel_tiers` must report the tier this architecture
    /// actually gets from `dispatch.rs::cpu_tier()`,
    /// not the dead `simd-neon`/`simd-avx2`/`simd-avx512` Cargo features
    /// (which have no `cfg(feature = "simd-...")` anywhere real).
    #[test]
    fn compiled_kernel_tiers_matches_target_arch_not_dead_features() {
        let tiers = compiled_kernel_tiers();
        assert!(tiers.contains(&"reference"));
        #[cfg(target_arch = "aarch64")]
        assert!(
            tiers.contains(&"neon"),
            "aarch64 always has NEON compiled in, regardless of the dead simd-neon feature"
        );
        #[cfg(target_arch = "x86_64")]
        {
            assert!(tiers.contains(&"avx2+fma"));
            assert!(tiers.contains(&"avx512f+bw+vl"));
        }
    }

    #[test]
    fn build_info_does_not_panic() {
        // Smoke test: build-info touches process/env/CPU-feature detection
        // and must never panic regardless of the environment it runs in.
        print_build_info();
    }
}

#[cfg(test)]
#[path = "model_desc_tests.rs"]
mod hybrid_report_tests;
