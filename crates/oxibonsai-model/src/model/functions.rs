//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(test)]
use crate::model::types::BonsaiModel;
#[cfg(test)]
use crate::model_registry::ModelVariant;
#[cfg(test)]
use oxibonsai_core::Qwen3Config;

// HOTFIX-TESTMEM: this module's tests used to build every fixture from
// `Qwen3Config::bonsai_8b()/bonsai_4b()/bonsai_1_7b()`. Constructing a
// `BonsaiModel` from one of those configs allocates ~5 GB of token_embd +
// output_weight tables (plus a ~1.2 GB KV cache for the 8B case) via
// `BonsaiModel::new` (model/types/mod.rs) — enough on its own to get a test
// process OOM-killed when several such tests run concurrently. None of the
// assertions below actually need production-sized weights:
//   - `model_creation` / `model_new_has_empty_blocks` / `model_reset_cache` /
//     `model_kv_cache_memory` only check that the passed-in config and the
//     empty-blocks/kv-cache bookkeeping are wired through correctly, which
//     holds for any config, so they now use `Qwen3Config::tiny_test()`.
//   - `model_variant_detection` and `model_info_methods` assert results that
//     depend on the *specific* real dimensions ((36, 4096), (24, 2560),
//     (28, 2048)) mapping to a known, non-`Custom` `ModelVariant` — see
//     `ModelVariant::from_config` (model_registry.rs). `tiny_test()`'s
//     (2, 64) doesn't match any known architecture, so swapping the config
//     without further changes would silently break `num_parameters() > 0`
//     and `model_size_bytes() > 0` (`ModelVariant::Custom::param_count()` and
//     `::expected_model_size_bytes()` are defined as exactly 0). Each is
//     therefore split into:
//     (a) a zero-allocation check of `ModelVariant::from_config`/the
//         `#[ignore]`d original full-size check (memory cost documented
//         inline), and
//     (b) a `_tiny_config`/`_tiny_is_custom` twin that runs by default and
//         exercises the same `BonsaiModel` accessor methods end to end
//         (`variant()`, `num_parameters()`, `model_size_bytes()`, ...) on a
//         cheap fixture, so the method-dispatch chain itself stays covered
//         even though the twin can't reach a named variant.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_creation() {
        let config = Qwen3Config::tiny_test();
        let model = BonsaiModel::new(config);
        assert_eq!(model.config().num_layers, 2);
        assert_eq!(model.config().hidden_size, 64);
    }

    #[test]
    fn model_new_has_empty_blocks() {
        let config = Qwen3Config::tiny_test();
        let model = BonsaiModel::new(config);
        assert_eq!(model.blocks.len(), 0);
    }

    /// Zero-allocation: `ModelVariant::from_config` is a pure function of
    /// `(num_layers, hidden_size)` (model_registry.rs) — exactly what
    /// `BonsaiModel::variant()` calls internally (with a non-ternary,
    /// non-FP8 `dominant_quant_type` for the config-only constructor, which
    /// leaves `from_config`'s result unchanged). Calling it directly proves
    /// the same detection the old test proved, without allocating the
    /// ~5 GB + ~1.2 GB (8B) / ~2.7 GB (4B) / ~1.3 GB (1.7B) each of
    /// `BonsaiModel::new(bonsai_8b()/bonsai_4b()/bonsai_1_7b())` would cost.
    #[test]
    fn model_variant_detection() {
        assert_eq!(
            ModelVariant::from_config(&Qwen3Config::bonsai_8b()),
            ModelVariant::Bonsai8B
        );
        assert_eq!(
            ModelVariant::from_config(&Qwen3Config::bonsai_4b()),
            ModelVariant::Bonsai4B
        );
        assert_eq!(
            ModelVariant::from_config(&Qwen3Config::bonsai_1_7b()),
            ModelVariant::Bonsai1_7B
        );
    }

    /// Tiny-config twin of `model_variant_detection`: exercises the real
    /// `BonsaiModel::variant()` method (not the free `from_config` function
    /// above) end to end on an actual instance, so the
    /// config -> model -> registry dispatch chain stays covered by a test
    /// that runs by default. `tiny_test()`'s (2, 64) doesn't match any known
    /// architecture, so detection correctly falls back to `Custom`.
    #[test]
    fn model_variant_detection_tiny_is_custom() {
        let model = BonsaiModel::new(Qwen3Config::tiny_test());
        assert_eq!(model.variant(), ModelVariant::Custom);
    }

    #[test]
    #[ignore = "constructs a full Bonsai-8B model (~5 GB token_embd + \
                output_weight tables plus a ~1.2 GB KV cache) solely to read \
                back accessor values; run explicitly with \
                `cargo test -- --ignored` to validate real-size model info. \
                See `model_info_methods_tiny_config` for the default-run twin."]
    fn model_info_methods() {
        let model = BonsaiModel::new(Qwen3Config::bonsai_8b());
        assert_eq!(model.num_layers(), 36);
        assert_eq!(model.hidden_size(), 4096);
        assert_eq!(model.context_length(), 65536);
        assert!(model.num_parameters() > 0);
        assert!(model.model_size_bytes() > 0);
    }

    /// Tiny-config twin of `model_info_methods` (kept `#[ignore]`d above
    /// because it needs a full production-size model to reach a non-`Custom`
    /// variant). `tiny_test()`'s (2, 64) maps to `ModelVariant::Custom`,
    /// whose `param_count`/`expected_model_size_bytes` are defined as
    /// exactly 0 (model_registry.rs), so this twin asserts the `Custom`
    /// contract on a ~78 MB fixture instead of the `Bonsai8B` one.
    #[test]
    fn model_info_methods_tiny_config() {
        let model = BonsaiModel::new(Qwen3Config::tiny_test());
        assert_eq!(model.num_layers(), 2);
        assert_eq!(model.hidden_size(), 64);
        assert_eq!(model.context_length(), 512);
        assert_eq!(
            model.num_parameters(),
            0,
            "Custom variant reports 0 parameters"
        );
        assert_eq!(
            model.model_size_bytes(),
            0,
            "Custom variant reports 0 model size bytes"
        );
    }

    #[test]
    fn model_reset_cache() {
        let mut model = BonsaiModel::new(Qwen3Config::tiny_test());
        model.reset_cache();
        assert_eq!(model.kv_cache_mut().seq_len(), 0);
    }

    #[test]
    fn model_kv_cache_memory() {
        let model = BonsaiModel::new(Qwen3Config::tiny_test());
        assert!(model.kv_cache_memory_bytes() > 0);
    }
}
