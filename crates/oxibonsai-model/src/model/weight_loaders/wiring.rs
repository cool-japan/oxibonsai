//! Load-time wiring for `BonsaiModel`: RoPE scaling (M-08) and the shape
//! invariants a bad configuration would otherwise break silently (M-12).
//!
//! Split out of `weight_loaders.rs` purely for file size; every item is
//! re-exported from there, so callers still say
//! `weight_loaders::build_rope_table`.

use oxibonsai_core::config::{Qwen3Config, RopeScaling};
use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::gguf::tensor_info::tensor_names;

use crate::error::{ModelError, ModelResult};
use crate::layers::rope::RopeTable;
use crate::layers::rope_scaling::RopeScalingStrategy;

// ─────────────────────────────────────────────────────────────────────────────
// RoPE scaling (M-08) — the config field becomes a real frequency table
// ─────────────────────────────────────────────────────────────────────────────

/// Turn [`Qwen3Config::rope_scaling`] into the strategy
/// [`RopeTable::new_with_scaling`] consumes (M-08).
///
/// `None` means "standard RoPE", which is what [`RopeTable::new`] builds.
///
/// **Why this exists.** `RopeScaling` (in `oxibonsai-core`) is the *parsed
/// declaration* — it carries exactly what `<arch>.rope.scaling.*` said, with
/// the optional keys still optional. `RopeScalingStrategy` (in
/// `crate::layers::rope_scaling`) is the *applied algorithm* — every
/// parameter resolved, ready to produce frequencies. This function is the
/// **only** production route from one to the other (the metadata-side
/// twin that used to exist, `RopeScalingStrategy::from_metadata`, is
/// deleted — this function's match arms mirror what it did, including its
/// `beta_fast = 32.0` / `beta_slow = 1.0` upstream defaults and its
/// case-insensitive type match) so a model loaded from a GGUF and a model
/// built from a hand-written [`Qwen3Config`] get the *same* table.
///
/// `RopeScaling::Other` is not a synonym for "unscaled": `RopeScaling`'s
/// parser buckets every non-`yarn` type there, including `linear` and the
/// dynamic-NTK spellings, which this crate can and does implement. Mapping
/// those to `None` would reproduce M-08's defect — a file that asks for
/// scaling silently running unscaled — one enum variant further along. Only a
/// genuinely unimplemented type yields `None`, and it warns when it does.
pub(crate) fn rope_scaling_strategy(config: &Qwen3Config) -> Option<RopeScalingStrategy> {
    match &config.rope_scaling {
        RopeScaling::None => None,
        RopeScaling::Yarn {
            factor,
            original_context_length,
            attn_factor,
            beta_fast,
            beta_slow,
        } => Some(RopeScalingStrategy::Yarn {
            original_max_position: *original_context_length as usize,
            factor: *factor,
            // Upstream (ggml / HF `transformers`) defaults, identical to the
            // ones `RopeScalingStrategy::from_metadata` applies.
            beta_fast: beta_fast.unwrap_or(32.0),
            beta_slow: beta_slow.unwrap_or(1.0),
            attn_factor: *attn_factor,
        }),
        RopeScaling::Other {
            scaling_type,
            factor,
            original_context_length,
        } => match scaling_type.to_ascii_lowercase().as_str() {
            "" | "none" => None,
            "linear" => Some(RopeScalingStrategy::Linear {
                scale_factor: factor.unwrap_or(1.0),
            }),
            "dynamic" | "ntk" | "dynamic-ntk" | "ntk-aware" => {
                Some(RopeScalingStrategy::DynamicNtk {
                    original_max_position: original_context_length.unwrap_or(0) as usize,
                    base: config.rope_freq_base,
                })
            }
            // A GGUF that spells the type with different case (`"YaRN"`,
            // `"Yarn"`) falls through `oxibonsai_core::config::RopeScaling`'s
            // case-sensitive match into `Other` instead of the dedicated
            // `RopeScaling::Yarn` variant, so it lands here rather than in
            // the arm above. Reachability regression fix: the retired
            // `RopeScalingStrategy::from_metadata` lowercased its type key
            // before matching and so accepted this case; without this arm a
            // mis-cased declaration would silently run unscaled. `Other`
            // does not carry `beta_fast`/`beta_slow`/`attn_factor` (only
            // `RopeScaling::Yarn` does), so this arm applies the same
            // upstream defaults the dedicated Yarn arm above falls back to
            // when those optional keys are themselves absent — it cannot
            // recover per-file overrides of them, but running YaRN with the
            // standard defaults is strictly better than running unscaled.
            "yarn" => Some(RopeScalingStrategy::Yarn {
                original_max_position: original_context_length.unwrap_or(0) as usize,
                factor: factor.unwrap_or(1.0),
                beta_fast: 32.0,
                beta_slow: 1.0,
                attn_factor: None,
            }),
            other => {
                tracing::warn!(
                    scaling_type = other,
                    "the model declares a RoPE scaling type this build does not implement; \
                     running unscaled RoPE, which will produce wrong output at long context"
                );
                None
            }
        },
    }
}

/// Build the RoPE table for `config`, honouring its declared scaling (M-08).
///
/// This is the only place a `BonsaiModel`'s `RopeTable` is constructed, so a
/// GGUF-loaded model, a config-only model and a model that *grew* its context
/// all get the same frequencies. `models/Bonsai-8B.gguf` declares
/// `qwen3.rope.scaling.{type=yarn, factor=4.0, original_context_length=16384}`,
/// and YaRN rescales `inv_freq` (and folds an attention-temperature `mscale`
/// into cos/sin) at **every** position — not only past the original context —
/// so a plain `RopeTable::new` here is wrong output on a shipped model, which
/// is exactly what M-08 reported.
///
/// # Errors
///
/// [`ModelError::RopeScaling`] when the declared parameters cannot produce a
/// table (a `factor < 1.0`, an odd `head_dim`, a LongRoPE factor-length
/// mismatch). Deliberately fatal: silently falling back to an unscaled table
/// makes "this build cannot honour the file" indistinguishable from "the file
/// asked for nothing".
pub(crate) fn build_rope_table(config: &Qwen3Config, max_seq_len: usize) -> ModelResult<RopeTable> {
    let strategy = rope_scaling_strategy(config);
    if let Some(strategy) = &strategy {
        tracing::debug!(
            head_dim = config.head_dim,
            max_seq_len,
            freq_base = config.rope_freq_base,
            ?strategy,
            "building a scaled RoPE table"
        );
    }
    Ok(RopeTable::new_with_scaling(
        config.head_dim,
        max_seq_len,
        config.rope_freq_base,
        strategy.as_ref(),
    )?)
}

/// [`build_rope_table`] for the **infallible** constructors.
///
/// [`crate::model::BonsaiModel::new`] and
/// [`crate::model::BonsaiModel::new_for_testing_with_blocks`] return `Self`,
/// not `ModelResult<Self>`, and are called from code this package does not
/// own, so they cannot become fallible. They get the scaled table whenever one
/// can be built and an unscaled one — with a `tracing::error!` naming the
/// reason — when it cannot. Every path that *can* report the failure
/// (`from_gguf*`, `grow_context`) uses [`build_rope_table`] and propagates it.
pub(crate) fn build_rope_table_or_unscaled(config: &Qwen3Config, max_seq_len: usize) -> RopeTable {
    match build_rope_table(config, max_seq_len) {
        Ok(table) => table,
        Err(e) => {
            tracing::error!(
                error = %e,
                head_dim = config.head_dim,
                "this configuration's RoPE scaling could not be applied; falling back to an \
                 unscaled table — long-context output will be wrong"
            );
            RopeTable::new(config.head_dim, max_seq_len, config.rope_freq_base)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Load-time shape invariants (M-12)
// ─────────────────────────────────────────────────────────────────────────────

/// Validate the shape invariants that must hold for a configuration *before*
/// any cache is allocated or any layer is built (M-12).
///
/// `crate::block::functions::validate_shapes` enforces the same relationships,
/// but only at the top of each `forward*` call — so a hand-built
/// [`Qwen3Config`], or one derived from a file whose metadata disagrees with
/// its tensors, survived construction and failed once per layer per token
/// instead. Two of these are hard crashes rather than wrong answers:
/// `num_kv_heads == 0` makes `num_heads / num_kv_heads` a divide-by-zero panic
/// and makes `KvCache::new` allocate a zero-length buffer first.
///
/// # Errors
///
/// [`ModelError::ShapeInvariant`] naming the relationship that failed.
pub(crate) fn validate_config_shapes(config: &Qwen3Config) -> ModelResult<()> {
    let invariant = |name: &str, expected: String, actual: String| ModelError::ShapeInvariant {
        tensor: name.to_string(),
        expected,
        actual,
    };
    if config.num_kv_heads == 0 {
        return Err(invariant(
            "config: attention.head_count_kv",
            "> 0 (it is the divisor of the GQA grouping and the KV-cache stride)".to_string(),
            "0".to_string(),
        ));
    }
    if config.num_attention_heads == 0 {
        return Err(invariant(
            "config: attention.head_count",
            "> 0".to_string(),
            "0".to_string(),
        ));
    }
    if !config
        .num_attention_heads
        .is_multiple_of(config.num_kv_heads)
    {
        return Err(invariant(
            "config: attention.head_count % attention.head_count_kv",
            format!(
                "0 (head_count {} must be a whole multiple of head_count_kv {})",
                config.num_attention_heads, config.num_kv_heads
            ),
            (config.num_attention_heads % config.num_kv_heads).to_string(),
        ));
    }
    if config.head_dim == 0 {
        return Err(invariant(
            "config: attention.key_length",
            "> 0".to_string(),
            "0".to_string(),
        ));
    }
    if config.hidden_size == 0 {
        return Err(invariant(
            "config: embedding_length",
            "> 0".to_string(),
            "0".to_string(),
        ));
    }
    Ok(())
}

/// Validate one layer's *tensor* shapes against the configuration the KV cache
/// will be allocated from (M-12).
///
/// GGUF stores a matrix as `[in_features, out_features]`, and every layer
/// shares one KV-cache stride derived from `config`. A layer whose `attn_k` /
/// `attn_v` is narrower or wider than `head_dim * num_kv_heads` therefore does
/// not error at the cache — it reads or writes a *different* layer's or head's
/// slot and returns plausible numbers. Checked here, once per layer at load,
/// so a mismatch is a load failure rather than silent cross-layer attention.
///
/// `attn_q` is allowed to be either `head_dim * num_heads` or exactly double
/// that: Bonsai 2's full-attention layers store `[q | gate]` interleaved per
/// head (design §3.2), the same allowance
/// `crate::block::functions::validate_shapes` makes.
pub(crate) fn validate_layer_tensor_shapes(
    gguf: &GgufFile<'_>,
    config: &Qwen3Config,
    layer_idx: usize,
) -> ModelResult<()> {
    let out_width = |suffix: &str| -> Option<usize> {
        let info = gguf
            .tensors
            .get(&tensor_names::block_tensor(layer_idx, suffix))?;
        // `[in, out]`; a 1-D tensor has no output width to check.
        info.shape.get(1).map(|&d| d as usize)
    };
    let check = |suffix: &str, actual: Option<usize>, allowed: &[usize]| -> ModelResult<()> {
        let Some(actual) = actual else {
            return Ok(());
        };
        if allowed.contains(&actual) {
            return Ok(());
        }
        Err(ModelError::ShapeInvariant {
            tensor: tensor_names::block_tensor(layer_idx, suffix),
            expected: allowed
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(" or "),
            actual: actual.to_string(),
        })
    };
    let q_plain = config.head_dim * config.num_attention_heads;
    let kv_width = config.head_dim * config.num_kv_heads;
    check(
        tensor_names::ATTN_Q,
        out_width(tensor_names::ATTN_Q),
        &[q_plain, q_plain * 2],
    )?;
    check(
        tensor_names::ATTN_K,
        out_width(tensor_names::ATTN_K),
        &[kv_width],
    )?;
    check(
        tensor_names::ATTN_V,
        out_width(tensor_names::ATTN_V),
        &[kv_width],
    )?;
    check(
        tensor_names::ATTN_OUTPUT,
        out_width(tensor_names::ATTN_OUTPUT),
        &[config.hidden_size],
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal, internally consistent config: one head, one KV head,
    /// `head_dim == hidden_size == 4`. Mirrors `weight_loaders::tests`'
    /// fixture of the same name, which is private to that module.
    fn tiny_qwen3_config() -> Qwen3Config {
        Qwen3Config {
            hidden_size: 4,
            intermediate_size: 4,
            num_layers: 1,
            num_attention_heads: 1,
            num_kv_heads: 1,
            head_dim: 4,
            value_length: 4,
            vocab_size: 8,
            max_context_length: 128,
            rms_norm_eps: 1e-6,
            rope_freq_base: 10_000.0,
            rope_scaling: RopeScaling::None,
            sliding_window: None,
            architecture: "qwen3".to_string(),
            model_name: "test".to_string(),
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // M-08 — `Qwen3Config::rope_scaling` becomes a real strategy
    // ══════════════════════════════════════════════════════════════════════

    fn config_with_scaling(scaling: RopeScaling) -> Qwen3Config {
        Qwen3Config {
            rope_scaling: scaling,
            ..tiny_qwen3_config()
        }
    }

    /// The `Bonsai-8B` declaration must convert to the YaRN strategy, with the
    /// upstream defaults filled in for the optional keys — the same
    /// `beta_fast = 32.0` / `beta_slow = 1.0` defaults `rope_scaling_strategy`
    /// applies everywhere else, so no caller can observe a divergence.
    #[test]
    fn yarn_declaration_converts_with_the_upstream_defaults() {
        let config = config_with_scaling(RopeScaling::Yarn {
            factor: 4.0,
            original_context_length: 16_384,
            attn_factor: None,
            beta_fast: None,
            beta_slow: None,
        });
        match rope_scaling_strategy(&config) {
            Some(RopeScalingStrategy::Yarn {
                original_max_position,
                factor,
                beta_fast,
                beta_slow,
                attn_factor,
            }) => {
                assert_eq!(original_max_position, 16_384);
                assert_eq!(factor, 4.0);
                assert_eq!(beta_fast, 32.0, "upstream beta_fast default");
                assert_eq!(beta_slow, 1.0, "upstream beta_slow default");
                assert_eq!(attn_factor, None, "None means 'derive the standard mscale'");
            }
            other => panic!("expected a YaRN strategy, got {other:?}"),
        }
    }

    /// Explicit optional keys must win over the defaults.
    #[test]
    fn yarn_declaration_keeps_explicit_optional_parameters() {
        let config = config_with_scaling(RopeScaling::Yarn {
            factor: 2.0,
            original_context_length: 4096,
            attn_factor: Some(1.25),
            beta_fast: Some(16.0),
            beta_slow: Some(2.0),
        });
        match rope_scaling_strategy(&config) {
            Some(RopeScalingStrategy::Yarn {
                beta_fast,
                beta_slow,
                attn_factor,
                ..
            }) => {
                assert_eq!(beta_fast, 16.0);
                assert_eq!(beta_slow, 2.0);
                assert_eq!(attn_factor, Some(1.25));
            }
            other => panic!("expected a YaRN strategy, got {other:?}"),
        }
    }

    /// `RopeScaling::Other` is NOT a synonym for "unscaled": the core parser
    /// buckets `linear` and the dynamic-NTK spellings there, and this crate
    /// implements both. Mapping them to `None` would reproduce M-08's defect
    /// one enum variant further along.
    #[test]
    fn other_scaling_types_this_build_implements_are_not_dropped() {
        let linear = config_with_scaling(RopeScaling::Other {
            scaling_type: "linear".to_string(),
            factor: Some(8.0),
            original_context_length: Some(4096),
        });
        assert!(matches!(
            rope_scaling_strategy(&linear),
            Some(RopeScalingStrategy::Linear { scale_factor }) if scale_factor == 8.0
        ));

        for spelling in ["dynamic", "ntk", "dynamic-ntk", "ntk-aware", "DYNAMIC"] {
            let cfg = config_with_scaling(RopeScaling::Other {
                scaling_type: spelling.to_string(),
                factor: Some(4.0),
                original_context_length: Some(2048),
            });
            match rope_scaling_strategy(&cfg) {
                Some(RopeScalingStrategy::DynamicNtk {
                    original_max_position,
                    base,
                }) => {
                    assert_eq!(original_max_position, 2048, "{spelling}");
                    assert_eq!(base, cfg.rope_freq_base, "{spelling}");
                }
                other => panic!("{spelling}: expected DynamicNtk, got {other:?}"),
            }
        }
    }

    /// A GGUF declaring a mis-cased `"YaRN"`/`"Yarn"` scaling type falls
    /// through `oxibonsai_core::config::RopeScaling::from_metadata`'s
    /// case-sensitive `"yarn"` match into the generic `Other` variant
    /// (`config.rs:177-203`), which does not carry `beta_fast`/`beta_slow`/
    /// `attn_factor`. `rope_scaling_strategy` must still resolve this to a
    /// `Yarn` strategy (using the upstream defaults for the fields `Other`
    /// cannot carry) rather than silently running unscaled RoPE — the exact
    /// reachability regression the deleted `RopeScalingStrategy::from_metadata`
    /// did not have (it lowercased its key before matching).
    #[test]
    fn mis_cased_yarn_type_in_other_still_resolves_to_yarn_strategy() {
        for spelling in ["YaRN", "Yarn", "YARN"] {
            let cfg = config_with_scaling(RopeScaling::Other {
                scaling_type: spelling.to_string(),
                factor: Some(4.0),
                original_context_length: Some(16_384),
            });
            assert_eq!(
                rope_scaling_strategy(&cfg),
                Some(RopeScalingStrategy::Yarn {
                    original_max_position: 16_384,
                    factor: 4.0,
                    beta_fast: 32.0,
                    beta_slow: 1.0,
                    attn_factor: None,
                }),
                "spelling {spelling}"
            );
        }
    }

    /// `None`, an explicit `"none"` and a type this build cannot run all mean
    /// "standard RoPE" — the last one loudly.
    #[test]
    fn unscaled_and_unimplemented_scaling_types_yield_no_strategy() {
        assert!(rope_scaling_strategy(&config_with_scaling(RopeScaling::None)).is_none());
        for spelling in ["", "none", "longrope", "su"] {
            let cfg = config_with_scaling(RopeScaling::Other {
                scaling_type: spelling.to_string(),
                factor: None,
                original_context_length: None,
            });
            assert!(rope_scaling_strategy(&cfg).is_none(), "{spelling}");
        }
    }

    /// `build_rope_table` must produce exactly `new_with_scaling`'s table, and
    /// something measurably different from the unscaled one.
    #[test]
    fn build_rope_table_applies_the_declared_scaling() {
        let config = config_with_scaling(RopeScaling::Yarn {
            factor: 4.0,
            original_context_length: 64,
            attn_factor: None,
            beta_fast: None,
            beta_slow: None,
        });
        let rows = 96usize;
        let built = build_rope_table(&config, rows).expect("valid YaRN parameters");
        let expected = RopeTable::new_with_scaling(
            config.head_dim,
            rows,
            config.rope_freq_base,
            rope_scaling_strategy(&config).as_ref(),
        )
        .expect("valid");
        let unscaled = RopeTable::new(config.head_dim, rows, config.rope_freq_base);
        let mut same = 0.0f32;
        let mut differ = 0.0f32;
        for pos in 0..rows {
            for ((b, e), u) in built
                .cos_at(pos)
                .iter()
                .zip(expected.cos_at(pos))
                .zip(unscaled.cos_at(pos))
            {
                same = same.max((b - e).abs());
                differ = differ.max((b - u).abs());
            }
        }
        assert_eq!(same, 0.0, "must equal `new_with_scaling`");
        assert!(differ > 1e-3, "must differ from the unscaled table");
    }

    /// An unusable declaration is a hard error on the fallible path and a
    /// logged degradation on the infallible one — never a silent success.
    #[test]
    fn an_impossible_scaling_declaration_is_reported_not_swallowed() {
        let config = config_with_scaling(RopeScaling::Yarn {
            // YaRN extends context, so a factor below 1.0 is meaningless.
            factor: 0.5,
            original_context_length: 64,
            attn_factor: None,
            beta_fast: None,
            beta_slow: None,
        });
        let err = build_rope_table(&config, 32).expect_err("factor < 1.0 must be rejected");
        assert_eq!(err.error_code(), "ROPE_SCALING", "{err}");
        // The infallible constructors' helper still returns a usable table.
        let fallback = build_rope_table_or_unscaled(&config, 32);
        assert_eq!(fallback.max_seq_len(), 32);
        assert_eq!(fallback.attention_scale(), 1.0, "unscaled fallback");
    }

    // ══════════════════════════════════════════════════════════════════════
    // M-12 — shape invariants are enforced at LOAD, not per token
    // ══════════════════════════════════════════════════════════════════════

    /// `num_kv_heads == 0` is the cheapest way to reach a hard crash: it makes
    /// `num_heads / num_kv_heads` a divide-by-zero in every block forward, and
    /// `KvCache::new` allocates a zero-length buffer first.
    #[test]
    fn a_zero_kv_head_count_is_rejected_before_anything_is_allocated() {
        let config = Qwen3Config {
            num_kv_heads: 0,
            ..tiny_qwen3_config()
        };
        let err = validate_config_shapes(&config).expect_err("0 KV heads must be rejected");
        assert_eq!(err.error_code(), "SHAPE_INVARIANT");
        assert!(
            err.to_string().contains("head_count_kv"),
            "the message must name the key: {err}"
        );
    }

    #[test]
    fn gqa_grouping_must_be_exact() {
        let config = Qwen3Config {
            num_attention_heads: 24,
            num_kv_heads: 5,
            ..tiny_qwen3_config()
        };
        let err = validate_config_shapes(&config).expect_err("24 % 5 != 0 must be rejected");
        assert_eq!(err.error_code(), "SHAPE_INVARIANT");
        assert!(err.to_string().contains("head_count"), "{err}");
        // The same numbers with an exact grouping are fine.
        assert!(validate_config_shapes(&Qwen3Config {
            num_attention_heads: 24,
            num_kv_heads: 4,
            ..tiny_qwen3_config()
        })
        .is_ok());
    }

    #[test]
    fn degenerate_dimensions_are_rejected() {
        for config in [
            Qwen3Config {
                num_attention_heads: 0,
                ..tiny_qwen3_config()
            },
            Qwen3Config {
                head_dim: 0,
                ..tiny_qwen3_config()
            },
            Qwen3Config {
                hidden_size: 0,
                ..tiny_qwen3_config()
            },
        ] {
            let err = validate_config_shapes(&config).expect_err("a zero dimension is invalid");
            assert_eq!(err.error_code(), "SHAPE_INVARIANT", "{err}");
        }
        assert!(validate_config_shapes(&tiny_qwen3_config()).is_ok());
    }

    /// A layer whose `attn_k` is not `head_dim * num_kv_heads` wide does not
    /// error at the KV cache — it reads another layer's or head's slot. Caught
    /// here instead, at load.
    #[test]
    fn a_layer_whose_kv_width_disagrees_with_the_cache_stride_is_rejected() {
        use oxibonsai_core::gguf::writer::{GgufWriter, TensorEntry, TensorType};
        use oxibonsai_core::MetadataWriteValue;

        let config = tiny_qwen3_config();
        let mut writer = GgufWriter::new();
        writer.add_metadata("general.name", MetadataWriteValue::Str("m".to_string()));
        let norm: Vec<u8> = [1.0f32; 4].iter().flat_map(|f| f.to_le_bytes()).collect();
        for name in [
            "blk.0.attn_norm.weight",
            "blk.0.ffn_norm.weight",
            "blk.0.attn_q_norm.weight",
            "blk.0.attn_k_norm.weight",
        ] {
            writer.add_tensor(TensorEntry {
                name: name.to_string(),
                shape: vec![4],
                tensor_type: TensorType::F32,
                data: norm.clone(),
            });
        }
        // `head_dim * num_kv_heads` is 4; declare 8.
        for (name, out) in [("blk.0.attn_q.weight", 4u64), ("blk.0.attn_k.weight", 8)] {
            writer.add_tensor(TensorEntry {
                name: name.to_string(),
                shape: vec![4, out],
                tensor_type: TensorType::F32,
                data: vec![0u8; 4 * out as usize * 4],
            });
        }
        let bytes = writer.to_bytes().expect("write fixture GGUF");
        let gguf = GgufFile::parse(&bytes).expect("parse fixture GGUF");

        let err = validate_layer_tensor_shapes(&gguf, &config, 0)
            .expect_err("a mismatched attn_k width must be rejected");
        assert_eq!(err.error_code(), "SHAPE_INVARIANT");
        assert!(
            err.to_string().contains("blk.0.attn_k.weight"),
            "the message must name the tensor: {err}"
        );
    }

    /// The Bonsai-2 `q|gate` interleave doubles `attn_q`'s width; that layout
    /// must stay acceptable so the check does not have to be relaxed later.
    #[test]
    fn a_doubled_attn_q_width_is_accepted_for_the_bonsai_2_q_gate_split() {
        use oxibonsai_core::gguf::writer::{GgufWriter, TensorEntry, TensorType};
        use oxibonsai_core::MetadataWriteValue;

        let config = tiny_qwen3_config();
        for width in [4u64, 8] {
            let mut writer = GgufWriter::new();
            writer.add_metadata("general.name", MetadataWriteValue::Str("m".to_string()));
            writer.add_tensor(TensorEntry {
                name: "blk.0.attn_q.weight".to_string(),
                shape: vec![4, width],
                tensor_type: TensorType::F32,
                data: vec![0u8; 4 * width as usize * 4],
            });
            let bytes = writer.to_bytes().expect("write fixture GGUF");
            let gguf = GgufFile::parse(&bytes).expect("parse fixture GGUF");
            assert!(
                validate_layer_tensor_shapes(&gguf, &config, 0).is_ok(),
                "attn_q width {width} must be accepted"
            );
        }
    }
}
