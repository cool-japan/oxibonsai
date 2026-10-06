//! Building the fixture: writing the GGUF file for a plan (metadata, the
//! optional `prism.hadamard.*` contract, every tensor), parsing it back
//! through the real reader for the configuration, and attaching the f64
//! reference forward. Also the negative fixture for the one configuration a
//! hybrid loader must refuse.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use oxibonsai_core::config_hybrid::HybridConfig;
use oxibonsai_core::error::{BonsaiError, BonsaiResult};
use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue, TensorEntry, TensorType};
use oxibonsai_core::hadamard_config::HadamardConfig;

use super::encode::encode_tensor;
use super::plan::{build_plan, FixturePlan, PlannedTensor};
use super::reference::{run_reference_forward, ReferenceForward};
use super::spec::{
    Dims, HybridFixtureSpec, BLOCK_COUNT, CONTEXT_LENGTH, FULL_ATTENTION_INTERVAL,
    HADAMARD_BLOCK_SIZE, HIDDEN, N_HEAD, N_KV, RMS_EPS, ROPE_DIM, ROPE_FREQ_BASE, ROPE_SECTIONS,
    SSM_CONV_KERNEL, SSM_TIME_STEP_RANK, VOCAB,
};

// ═════════════════════════════════════════════════════════════════════════
// 10. The built fixture
// ═════════════════════════════════════════════════════════════════════════

static FIXTURE_COUNTER: AtomicU64 = AtomicU64::new(0);

fn unique_temp_path(tag: &str, seed: u64) -> PathBuf {
    let n = FIXTURE_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "oxibonsai_hybrid_fixture_{tag}_{}_{seed}_{n}.gguf",
        std::process::id()
    ))
}

/// A generated hybrid GGUF fixture: the file on disk, its parsed
/// configuration, the exact pre-quantization weight values (for
/// byte-fidelity checks), and the independent f64 reference forward.
pub struct HybridFixture {
    pub path: PathBuf,
    pub spec: HybridFixtureSpec,
    pub dims: Dims,
    pub cfg: HybridConfig,
    pub hadamard: Option<HadamardConfig>,
    tensors: BTreeMap<String, PlannedTensor>,
    pub reference: ReferenceForward,
}

impl HybridFixture {
    /// The exact row-major pre-quantization `f32` values written for
    /// tensor `name` (already bf16-rounded for `ssm_alpha`/`ssm_beta`), or
    /// `None` if no such tensor was planned.
    pub fn planned_values(&self, name: &str) -> Option<&[f32]> {
        self.tensors.get(name).map(|t| t.values.as_slice())
    }

    pub fn planned_shape(&self, name: &str) -> Option<&[u64]> {
        self.tensors.get(name).map(|t| t.shape.as_slice())
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(|s| s.as_str())
    }
}

impl Drop for HybridFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// A metadata-only fixture for the one configuration design §3.3 says a
/// hybrid loader must *refuse*: `gdn_v_grouped == false` (columns folded
/// tiled) with `v_per_k > 1` (an ungrouped fold cannot be re-derived from
/// a tiled read). No `ReferenceForward` is computed — there is no correct
/// answer for a configuration the spec says to reject, so this only
/// carries what `build()` and `parse()` need to exist and be consistent.
pub struct InvalidUngroupedFixture {
    pub path: PathBuf,
    pub v_per_k: usize,
}

impl Drop for InvalidUngroupedFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Build the negative fixture described on [`InvalidUngroupedFixture`]:
/// hadamard on (`gdn_v_grouped` only exists as a key at all when the
/// Hadamard contract is present), but `gdn_v_grouped = false` in the
/// metadata while the weight layout still uses `n_k_heads = 2` (`v_per_k ==
/// 2`) — exactly the combination the design says the hybrid loader must
/// reject.
pub fn build_invalid_ungrouped_fixture(
    seed: u64,
    quant: TensorType,
) -> BonsaiResult<InvalidUngroupedFixture> {
    // `gdn_v_grouped: true` here only selects `n_k_heads == 2` (`v_per_k ==
    // 2`) via `Dims::from_spec`; the metadata's own `gdn_v_grouped` value
    // is overridden to `false` below, producing the inconsistent pair.
    let spec = HybridFixtureSpec {
        quant,
        hadamard: true,
        gdn_v_grouped: true,
        seed,
        hidden: HIDDEN,
        hadamard_block: HADAMARD_BLOCK_SIZE,
    };
    let plan = build_plan(&spec);
    let path = unique_temp_path("invalid_ungrouped", seed);
    write_gguf(&spec, &plan, &path, false)?;
    Ok(InvalidUngroupedFixture {
        path,
        v_per_k: plan.dims.v_per_k(),
    })
}

/// Build one hybrid GGUF fixture and its embedded f64 reference forward.
pub fn build(spec: &HybridFixtureSpec) -> BonsaiResult<HybridFixture> {
    let plan = build_plan(spec);
    let path = unique_temp_path("hybrid", spec.seed);
    write_gguf(spec, &plan, &path, spec.gdn_v_grouped)?;

    let bytes = std::fs::read(&path).map_err(BonsaiError::MmapError)?;
    let file = oxibonsai_core::gguf::reader::GgufFile::parse(&bytes)?;
    let cfg = HybridConfig::from_metadata(&file.metadata)?;
    let hadamard = HadamardConfig::from_metadata(&file.metadata)?;
    if let Some(had) = &hadamard {
        had.validate_against_tensors(&file.tensors)?;
    }

    let reference = run_reference_forward(spec, &plan);

    Ok(HybridFixture {
        path,
        spec: *spec,
        dims: plan.dims,
        cfg,
        hadamard,
        tensors: plan.tensors,
        reference,
    })
}

fn write_gguf(
    spec: &HybridFixtureSpec,
    plan: &FixturePlan,
    path: &std::path::Path,
    gdn_v_grouped_metadata: bool,
) -> BonsaiResult<()> {
    let mut w = GgufWriter::new();
    w.add_metadata(
        "general.architecture",
        MetadataWriteValue::Str("qwen35".to_string()),
    );
    w.add_metadata(
        "general.name",
        MetadataWriteValue::Str("hybrid-fixture".to_string()),
    );
    w.add_metadata(
        "qwen35.embedding_length",
        MetadataWriteValue::U32(plan.dims.hidden as u32),
    );
    w.add_metadata(
        "qwen35.block_count",
        MetadataWriteValue::U32(BLOCK_COUNT as u32),
    );
    w.add_metadata(
        "qwen35.attention.head_count",
        MetadataWriteValue::U32(N_HEAD as u32),
    );
    w.add_metadata(
        "qwen35.attention.head_count_kv",
        MetadataWriteValue::U32(N_KV as u32),
    );
    w.add_metadata(
        "qwen35.attention.key_length",
        MetadataWriteValue::U32(plan.dims.head_dim as u32),
    );
    w.add_metadata(
        "qwen35.attention.value_length",
        MetadataWriteValue::U32(plan.dims.head_dim as u32),
    );
    w.add_metadata(
        "qwen35.feed_forward_length",
        MetadataWriteValue::U32(plan.dims.ffn as u32),
    );
    w.add_metadata("qwen35.vocab_size", MetadataWriteValue::U32(VOCAB as u32));
    w.add_metadata(
        "qwen35.context_length",
        MetadataWriteValue::U32(CONTEXT_LENGTH as u32),
    );
    w.add_metadata(
        "qwen35.attention.layer_norm_rms_epsilon",
        MetadataWriteValue::F32(RMS_EPS as f32),
    );
    w.add_metadata(
        "qwen35.rope.freq_base",
        MetadataWriteValue::F32(ROPE_FREQ_BASE as f32),
    );
    w.add_metadata(
        "qwen35.full_attention_interval",
        MetadataWriteValue::U32(FULL_ATTENTION_INTERVAL as u32),
    );
    w.add_metadata(
        "qwen35.rope.dimension_count",
        MetadataWriteValue::U32(ROPE_DIM as u32),
    );
    w.add_metadata(
        "qwen35.rope.dimension_sections",
        MetadataWriteValue::ArrayI32(ROPE_SECTIONS.to_vec()),
    );
    w.add_metadata(
        "qwen35.ssm.conv_kernel",
        MetadataWriteValue::U32(SSM_CONV_KERNEL as u32),
    );
    w.add_metadata(
        "qwen35.ssm.state_size",
        MetadataWriteValue::U32(plan.dims.head_k_dim as u32),
    );
    w.add_metadata(
        "qwen35.ssm.group_count",
        MetadataWriteValue::U32(plan.dims.n_k_heads as u32),
    );
    w.add_metadata(
        "qwen35.ssm.time_step_rank",
        MetadataWriteValue::U32(SSM_TIME_STEP_RANK as u32),
    );
    w.add_metadata(
        "qwen35.ssm.inner_size",
        MetadataWriteValue::U32(plan.dims.inner_size as u32),
    );
    w.add_metadata("general.sampling.temp", MetadataWriteValue::F32(1.0));
    w.add_metadata("general.sampling.top_p", MetadataWriteValue::F32(0.95));
    w.add_metadata("general.sampling.top_k", MetadataWriteValue::U32(20));

    // ── id-42 disambiguation metadata (design §7.2 "id-42 resolution"
    // row): `general.quantization_version` is the only
    // real discriminator between the on-disk readings of wire id 42
    // (`oxibonsai_core::gguf::quant_resolve` module docs) — the mainline
    // `Q2_0_g64` reading carries the numeric `u32 2` PrismML files use,
    // while OxiBonsai's own legacy `TQ2_0_g128` writer emits the **string**
    // `"TQ2_0_G128"` instead. Give each wire-id-42 fixture variant its real
    // spelling so `resolve_type_42_with_sample` can be exercised against
    // these fixtures, not only against real model files. Other quant
    // formats (F32/PQ2_0/PTQ1_0/Q1_0_g128) are never ambiguous at the
    // wire-id level, so this key is left unset for them.
    match spec.quant {
        TensorType::Q2_0G64 => {
            w.add_metadata("general.quantization_version", MetadataWriteValue::U32(2));
        }
        TensorType::TQ2_0_g128 => {
            w.add_metadata(
                "general.quantization_version",
                MetadataWriteValue::Str("TQ2_0_G128".to_string()),
            );
        }
        _ => {}
    }

    // ── Tiny tokenizer block (enough for HybridConfig/GgufFile to parse a
    // "complete" file; the tokenizer loader itself is a different
    // package's scope) ─────────────────────────────────────────────────
    w.add_metadata(
        "tokenizer.ggml.model",
        MetadataWriteValue::Str("gpt2".to_string()),
    );
    let tokens: Vec<String> = (0..VOCAB).map(|i| format!("<tok{i}>")).collect();
    w.add_metadata(
        "tokenizer.ggml.tokens",
        MetadataWriteValue::ArrayStr(tokens),
    );
    w.add_metadata(
        "tokenizer.ggml.eos_token_id",
        MetadataWriteValue::U32((VOCAB - 1) as u32),
    );
    w.add_metadata("tokenizer.ggml.bos_token_id", MetadataWriteValue::U32(0));
    w.add_metadata(
        "tokenizer.ggml.add_bos_token",
        MetadataWriteValue::Bool(false),
    );

    // ── Hadamard contract ────────────────────────────────────────────────
    if spec.hadamard {
        w.add_metadata("prism.hadamard.version", MetadataWriteValue::U32(1));
        w.add_metadata(
            "prism.hadamard.block_size",
            MetadataWriteValue::U32(plan.dims.hadamard_block as u32),
        );
        w.add_metadata(
            "prism.hadamard.transform",
            MetadataWriteValue::Str("normalized-sylvester-walsh-hadamard".to_string()),
        );
        w.add_metadata(
            "prism.hadamard.axis",
            MetadataWriteValue::Str("input-last-dimension".to_string()),
        );
        w.add_metadata(
            "prism.hadamard.sign_mode",
            MetadataWriteValue::Str("explicit".to_string()),
        );
        let widths: Vec<i32> = plan.signs.keys().map(|&w| w as i32).collect();
        let mut values: Vec<i32> = Vec::new();
        for width in plan.signs.keys() {
            for &s in &plan.signs[width] {
                values.push(s as i32);
            }
        }
        w.add_metadata(
            "prism.hadamard.sign_widths",
            MetadataWriteValue::ArrayI32(widths),
        );
        w.add_metadata(
            "prism.hadamard.sign_values",
            MetadataWriteValue::ArrayI32(values),
        );
        w.add_metadata(
            "prism.hadamard.weight_names",
            MetadataWriteValue::ArrayStr(plan.folded_names.clone()),
        );
        w.add_metadata(
            "prism.hadamard.inverse_weight_names",
            MetadataWriteValue::ArrayStr(vec!["token_embd.weight".to_string()]),
        );
        w.add_metadata(
            "prism.hadamard.gdn_v_grouped",
            MetadataWriteValue::Bool(gdn_v_grouped_metadata),
        );
    }

    for (name, planned) in &plan.tensors {
        let (wire_type, bytes) = encode_tensor(planned.kind, spec.quant, &planned.values)?;
        w.add_tensor(TensorEntry {
            name: name.clone(),
            shape: planned.shape.clone(),
            tensor_type: wire_type,
            data: bytes,
        });
    }

    let mut file = std::fs::File::create(path).map_err(BonsaiError::MmapError)?;
    w.write(&mut file).map_err(|e| BonsaiError::KQuantError {
        reason: format!("hybrid fixture write failed: {e}"),
    })?;
    Ok(())
}
