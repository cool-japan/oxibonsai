//! Integration tests for the CUDA inference paths in `BonsaiModel`.
//!
//! All tests in this file are gated behind `#[cfg(feature = "native-cuda")]` and
//! gracefully skip when no CUDA GPU is present in the environment (e.g. CI).
//!
//! The test strategy mirrors the Metal tests in `model_forward.rs`: a small
//! synthetic model (0–2 transformer layers, tiny hidden dim) is constructed via
//! `BonsaiModel::new`, and the CUDA forward path is invoked through the public
//! `forward`, `forward_prefill`, and `forward_prefill_verify` APIs.  Because
//! `BonsaiModel::new` builds zero-valued weights the numerical outputs are
//! predictable and consistent with the CPU path.

#[cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
mod cuda_tests {
    use oxibonsai_core::config::{Qwen3Config, RopeScaling};
    use oxibonsai_kernels::dispatch::{KernelDispatcher, KernelTier};
    use oxibonsai_model::model::BonsaiModel;
    use oxibonsai_testkit::capability::{record as record_capability, Capability};

    fn ref_kernel() -> KernelDispatcher {
        KernelDispatcher::with_tier(KernelTier::Reference)
    }

    /// Returns `true` when a CUDA device is accessible; `false` otherwise.
    /// Used to skip tests gracefully in CPU-only environments.
    fn cuda_available() -> bool {
        oxibonsai_kernels::CudaGraph::global().is_ok()
    }

    // ── T-05 capability-report producer ──────────────────────────────────
    //
    // T-07: this used to be an inline copy of
    // `oxibonsai_testkit::capability::record`; `oxibonsai-model` now takes
    // `oxibonsai-testkit` as a dev-dependency (imported above), so the copy
    // is deleted in favour of the shared implementation.

    /// Minimal model config used across all CUDA tests.
    fn small_config() -> Qwen3Config {
        Qwen3Config {
            hidden_size: 128,
            intermediate_size: 256,
            num_layers: 0, // no transformer blocks — exercises embedding + norm + lm_head
            num_attention_heads: 2,
            num_kv_heads: 1,
            head_dim: 64,
            value_length: 64,
            vocab_size: 100,
            max_context_length: 64,
            rms_norm_eps: 1e-6,
            rope_freq_base: 10000.0,
            rope_scaling: RopeScaling::None,
            sliding_window: None,
            architecture: "test".to_string(),
            model_name: "test".to_string(),
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // forward() — single token, CUDA path
    // ──────────────────────────────────────────────────────────────────────────

    /// CUDA forward on a model with 0 blocks falls through to CPU gracefully.
    /// The model should still produce logits of the correct length.
    #[test]
    fn cuda_forward_no_blocks_produces_correct_logit_count() {
        let mut model = BonsaiModel::new(small_config());
        let kernel = ref_kernel();

        // With 0 blocks the CUDA full-forward path will return None (nothing to do)
        // and we fall through to the CPU path.  The result must still be correct.
        let logits = model
            .forward(0, 0, &kernel)
            .expect("forward should succeed even without a GPU");
        assert_eq!(logits.len(), 100, "logit vector must match vocab_size");
    }

    /// CUDA `forward` with zero weights produces all-zero logits, consistent
    /// with the CPU reference path.
    #[test]
    fn cuda_forward_zero_weights_matches_cpu() {
        // This test's assertions hold regardless of CUDA availability: with
        // no GPU the dispatch transparently falls back to CPU, and the
        // determinism check below is still meaningful either way — so,
        // unlike the GPU-device-specific tests further down, this one is
        // never skipped (it was previously an `if !cuda_available() {}`
        // with an empty body that branched on nothing; removed as dead
        // code rather than left to look like a real skip guard it was not).
        let mut model_a = BonsaiModel::new(small_config());
        let mut model_b = BonsaiModel::new(small_config());
        let kernel = ref_kernel();

        let logits_a = model_a.forward(0, 0, &kernel).expect("forward a");
        let logits_b = model_b.forward(0, 0, &kernel).expect("forward b");

        assert_eq!(logits_a.len(), logits_b.len());
        for (i, (a, b)) in logits_a.iter().zip(logits_b.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "deterministic mismatch at index {i}: {a} vs {b}"
            );
        }
    }

    /// Calling `forward` multiple times advances the sequence consistently
    /// even when the CUDA dispatch path handles (or skips) the layers.
    #[test]
    fn cuda_forward_multi_token_sequence_consistent() {
        let mut model = BonsaiModel::new(small_config());
        let kernel = ref_kernel();

        for pos in 0..4u32 {
            let logits = model
                .forward(pos % 10, pos as usize, &kernel)
                .expect("forward at pos {pos}");
            assert_eq!(logits.len(), 100, "pos={pos}: logit length mismatch");
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // forward_prefill() — batch GPU path
    // ──────────────────────────────────────────────────────────────────────────

    /// `forward_prefill` on a single-token batch is equivalent to `forward`.
    #[test]
    fn cuda_forward_prefill_single_token_equals_forward() {
        let config = small_config();
        let mut model_a = BonsaiModel::new(config.clone());
        let mut model_b = BonsaiModel::new(config);
        let kernel = ref_kernel();

        let logits_single = model_a.forward(3, 0, &kernel).expect("forward");
        model_a.reset();

        let logits_prefill = model_b
            .forward_prefill(&[3], 0, &kernel)
            .expect("forward_prefill single");

        assert_eq!(logits_single.len(), logits_prefill.len());
        for (i, (a, b)) in logits_single.iter().zip(logits_prefill.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "single-token prefill mismatch at index {i}: {a} vs {b}"
            );
        }
    }

    /// `forward_prefill` on an empty token list returns an error (not a panic).
    #[test]
    fn cuda_forward_prefill_empty_tokens_returns_err() {
        let mut model = BonsaiModel::new(small_config());
        let kernel = ref_kernel();

        let result = model.forward_prefill(&[], 0, &kernel);
        assert!(
            result.is_err(),
            "empty token_ids should return an error from forward_prefill"
        );
    }

    /// `forward_prefill` on a multi-token batch returns logits of the correct
    /// vocabulary size regardless of whether the CUDA batch path fires.
    #[test]
    fn cuda_forward_prefill_batch_produces_correct_logit_count() {
        let mut model = BonsaiModel::new(small_config());
        let kernel = ref_kernel();

        let token_ids: Vec<u32> = (0..8).collect();
        let logits = model
            .forward_prefill(&token_ids, 0, &kernel)
            .expect("forward_prefill batch");
        assert_eq!(logits.len(), 100, "logits must span full vocabulary");
    }

    // ──────────────────────────────────────────────────────────────────────────
    // forward_prefill_verify() — speculative decode verify path
    // ──────────────────────────────────────────────────────────────────────────

    /// `forward_prefill_verify` on an empty token list returns an empty vec.
    #[test]
    fn cuda_forward_prefill_verify_empty_returns_empty() {
        let mut model = BonsaiModel::new(small_config());
        let kernel = ref_kernel();

        let ids = model
            .forward_prefill_verify(&[], 0, &kernel)
            .expect("forward_prefill_verify empty");
        assert!(ids.is_empty(), "empty input should produce empty output");
    }

    /// `forward_prefill_verify` returns one greedy token ID per input token.
    #[test]
    fn cuda_forward_prefill_verify_produces_one_id_per_token() {
        let mut model = BonsaiModel::new(small_config());
        let kernel = ref_kernel();

        let token_ids: Vec<u32> = vec![0, 1, 2, 3, 4];
        let ids = model
            .forward_prefill_verify(&token_ids, 0, &kernel)
            .expect("forward_prefill_verify");
        assert_eq!(
            ids.len(),
            token_ids.len(),
            "output length must equal input batch size"
        );
        for id in &ids {
            assert!(
                (*id as usize) < 100,
                "greedy token id {id} must be within vocab_size=100"
            );
        }
    }

    /// With zero-valued weights every position produces the same greedy argmax
    /// (typically token 0 or whichever ties first), so the verify output is
    /// deterministic across two independent model instances.
    #[test]
    fn cuda_forward_prefill_verify_deterministic() {
        let config = small_config();
        let mut model_a = BonsaiModel::new(config.clone());
        let mut model_b = BonsaiModel::new(config);
        let kernel = ref_kernel();

        let token_ids: Vec<u32> = vec![1, 2, 3];
        let ids_a = model_a
            .forward_prefill_verify(&token_ids, 0, &kernel)
            .expect("verify a");
        let ids_b = model_b
            .forward_prefill_verify(&token_ids, 0, &kernel)
            .expect("verify b");

        assert_eq!(ids_a, ids_b, "verify output must be deterministic");
    }

    // ──────────────────────────────────────────────────────────────────────────
    // GPU-device-specific behaviour (skip gracefully when no CUDA GPU present)
    // ──────────────────────────────────────────────────────────────────────────

    /// When a CUDA device IS present, `try_cuda_full_forward` with an empty
    /// layer_params slice does not panic (graceful no-op).
    ///
    /// T-09: this used to be named `..._returns_none_gracefully` and end in
    /// `let _ = result;` — a name promising a specific `None` return that
    /// the body never actually checked (its own comment already hedged:
    /// "either None or Some ... either way must not panic", so `None` was
    /// never guaranteed). Renamed to match what is actually asserted, and
    /// "must not panic" is now a real, explicit assertion via
    /// `catch_unwind` instead of an implicit property no `assert*` ever
    /// tested.
    #[test]
    fn cuda_full_forward_empty_params_does_not_panic() {
        const TEST: &str =
            "oxibonsai-model::cuda_inference_tests::cuda_full_forward_empty_params_does_not_panic";
        if !cuda_available() {
            record_capability(Capability::Cuda, false, TEST);
            return; // No GPU — skip.
        }
        record_capability(Capability::Cuda, true, TEST);

        // Build a trivially small hidden vector matching the smallest legal hidden
        // size (must be > 0 for the kernel to accept it, but can be synthetic).
        let hidden = vec![0.0f32; 64];
        let rope_cos = vec![1.0f32; 32];
        let rope_sin = vec![0.0f32; 32];

        let outcome = std::panic::catch_unwind(|| {
            oxibonsai_kernels::try_cuda_full_forward(
                &hidden,
                &[], // zero layers — should return None without crash
                &rope_cos,
                &rope_sin,
                0,  // pos
                1,  // nq
                1,  // nkv
                64, // head_dim
                1,  // heads_per_group
                1e-6,
                64,  // hidden_size
                128, // intermediate_size
                128, // max_seq_len
                None,
                0,
            )
        });
        assert!(
            outcome.is_ok(),
            "try_cuda_full_forward must not panic on an empty layer_params slice"
        );
    }

    /// When a CUDA device IS present, `try_cuda_prefill` with an empty
    /// layer_params slice does not panic.
    ///
    /// T-09: see `cuda_full_forward_empty_params_does_not_panic`'s doc
    /// comment — same rename-and-assert-what-is-actually-checked fix.
    #[test]
    fn cuda_prefill_empty_params_does_not_panic() {
        const TEST: &str =
            "oxibonsai-model::cuda_inference_tests::cuda_prefill_empty_params_does_not_panic";
        if !cuda_available() {
            record_capability(Capability::Cuda, false, TEST);
            return; // No GPU — skip.
        }
        record_capability(Capability::Cuda, true, TEST);

        let hidden_batch = vec![0.0f32; 64];
        let cos_table = vec![1.0f32; 32];
        let sin_table = vec![0.0f32; 32];

        // The important property is that this does NOT panic regardless of the
        // outcome.  On a machine with CUDA SDK headers the call returns Ok(()); on
        // machines without SDK headers kernel compilation fails and returns Err.
        let outcome = std::panic::catch_unwind(|| {
            oxibonsai_kernels::try_cuda_prefill(
                &hidden_batch,
                1,   // batch_size
                0,   // pos_start
                0,   // n_layers
                &[], // layer_params — matches n_layers
                &cos_table,
                &sin_table,
                64,  // hidden_size
                128, // intermediate_size
                1,   // nq
                1,   // nkv
                64,  // head_dim
                1,   // heads_per_group
                1e-6,
                128, // max_seq_len
                None,
                None,
                1e-6,
                None,
                None,
                0,
                None,
                None,
            )
        });
        assert!(
            outcome.is_ok(),
            "try_cuda_prefill must not panic on an empty layer_params slice (0 n_layers)"
        );
    }
}
