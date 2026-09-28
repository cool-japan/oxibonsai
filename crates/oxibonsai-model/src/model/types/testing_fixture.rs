//! `BonsaiModel::new_for_testing_with_blocks`: a tiny, deterministic model
//! with real Transformer blocks for the prefix-cache tests. Split out of
//! `model/types/mod.rs` (B2-11-FIX).

use super::constructors::{effective_context, force_cpu_decode_after_from_env, prealloc_context};
use super::embedding::EmbeddingTable;
use super::{BonsaiModel, ModelScratch, OutputWeight, MAX_PREALLOC_CONTEXT};
use crate::kv_cache::{KvCache, GROWTH_CHUNK_POSITIONS};
use crate::layers::rms_norm::RmsNorm;
use crate::model::weight_loaders::build_rope_table_or_unscaled;
use oxibonsai_core::config::Qwen3Config;

impl BonsaiModel<'static> {
    /// Build a model with real (but tiny, deterministic) Transformer blocks
    /// for testing the prefix-cache path.
    ///
    /// Unlike [`BonsaiModel::new`] (which leaves `blocks` empty), this
    /// constructor instantiates `config.num_layers` real
    /// [`crate::block::TransformerBlock`]s backed by leaked weight
    /// allocations. The leaked memory is acceptable in tests, where the
    /// process is short-lived. The resulting model writes its KV cache via
    /// the standard CPU forward path, allowing the prefix cache to be
    /// exercised end-to-end.
    pub fn new_for_testing_with_blocks(config: Qwen3Config) -> Self {
        use crate::block::TransformerBlock;
        use crate::layers::linear::{Linear1Bit, LinearLayer};
        use half::f16;
        use oxibonsai_core::tensor::BlockQ1_0G128;
        use oxibonsai_kernels::{KernelDispatcher, KernelTier};
        use std::sync::Arc;

        let h = config.hidden_size;
        let hd = config.head_dim;
        let nq = config.num_attention_heads;
        let nkv = config.num_kv_heads;
        let inter = config.intermediate_size;

        // Q1_0_g128 packs 128 weights per block → blocks_per_in_row = in_features / 128.
        // We require in_features % 128 == 0.
        assert!(
            h.is_multiple_of(128),
            "test fixture requires hidden_size to be a multiple of 128"
        );
        assert!(
            inter.is_multiple_of(128),
            "test fixture requires intermediate_size to be a multiple of 128"
        );

        let h_bpr = h / 128;
        let inter_bpr = inter / 128;

        // Force the Reference (CPU) tier so the CPU `KvCache` is populated by
        // the forward path. With auto_detect on a GPU host the dispatcher
        // would route through Metal/CUDA, leaving the CPU cache empty and
        // breaking prefix-cache tests that round-trip through it.
        let kernel_arc = Arc::new(KernelDispatcher::with_tier(KernelTier::Reference));
        let max_context = effective_context(&config, None, None);
        let prealloc = prealloc_context(max_context, Some(MAX_PREALLOC_CONTEXT));
        // The host-KV default every constructor shares (lazy `f16`, limit =
        // effective context), starting at one growth chunk.
        let kv_cache = KvCache::new_lazy_f16(
            config.num_layers,
            config.num_kv_heads,
            config.head_dim,
            max_context,
            prealloc.min(GROWTH_CHUNK_POSITIONS),
        );
        let rope = build_rope_table_or_unscaled(&config, prealloc);

        // Helper: build a leaked Vec<BlockQ1_0G128> with deterministic data so
        // the test fixture is reproducible. Returns a 'static slice.
        fn make_blocks_static(n: usize, scale: f32, pattern: u8) -> &'static [BlockQ1_0G128] {
            let v: Vec<BlockQ1_0G128> = (0..n)
                .map(|i| BlockQ1_0G128 {
                    d: f16::from_f32(scale),
                    qs: [pattern.wrapping_add((i & 0xff) as u8); 16],
                })
                .collect();
            // Leak the allocation so the slice lives for 'static. Acceptable in tests.
            Box::leak(v.into_boxed_slice())
        }

        let mut blocks = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            // Per-block weight allocations (leaked).
            let q_blk = make_blocks_static(nq * hd * h_bpr, 0.01, 0xA5);
            let k_blk = make_blocks_static(nkv * hd * h_bpr, 0.01, 0x5A);
            let v_blk = make_blocks_static(nkv * hd * h_bpr, 0.01, 0x33);
            let o_blk = make_blocks_static(h * (nq * hd / 128).max(1), 0.01, 0xCC);
            let g_blk = make_blocks_static(inter * h_bpr, 0.01, 0x77);
            let u_blk = make_blocks_static(inter * h_bpr, 0.01, 0x88);
            let d_blk = make_blocks_static(h * inter_bpr, 0.01, 0x99);

            let attn_q: LinearLayer<'static> =
                Linear1Bit::new(q_blk, nq * hd, h, kernel_arc.clone())
                    .expect("q proj")
                    .into();
            let attn_k: LinearLayer<'static> =
                Linear1Bit::new(k_blk, nkv * hd, h, kernel_arc.clone())
                    .expect("k proj")
                    .into();
            let attn_v: LinearLayer<'static> =
                Linear1Bit::new(v_blk, nkv * hd, h, kernel_arc.clone())
                    .expect("v proj")
                    .into();
            let attn_out: LinearLayer<'static> =
                Linear1Bit::new(o_blk, h, nq * hd, kernel_arc.clone())
                    .expect("o proj")
                    .into();
            let ffn_gate: LinearLayer<'static> =
                Linear1Bit::new(g_blk, inter, h, kernel_arc.clone())
                    .expect("gate proj")
                    .into();
            let ffn_up: LinearLayer<'static> = Linear1Bit::new(u_blk, inter, h, kernel_arc.clone())
                .expect("up proj")
                .into();
            let ffn_down: LinearLayer<'static> =
                Linear1Bit::new(d_blk, h, inter, kernel_arc.clone())
                    .expect("down proj")
                    .into();

            let block = TransformerBlock::new(
                layer_idx,
                RmsNorm::new(vec![1.0; h], config.rms_norm_eps),
                attn_q,
                attn_k,
                attn_v,
                attn_out,
                RmsNorm::new(vec![1.0; hd], config.rms_norm_eps),
                RmsNorm::new(vec![1.0; hd], config.rms_norm_eps),
                RmsNorm::new(vec![1.0; h], config.rms_norm_eps),
                ffn_gate,
                ffn_up,
                ffn_down,
                nq,
                nkv,
                hd,
                h,
            );
            blocks.push(block);
        }

        Self {
            // The fixture's embedding is a constant 0.01 in every element, as
            // before — now synthesized per row instead of materialized.
            token_embd: EmbeddingTable::constant(0.01, config.vocab_size, h),
            shared_embd: std::sync::Arc::from(Vec::new()),
            blocks,
            output_norm: RmsNorm::new(vec![1.0; h], config.rms_norm_eps),
            output_weight: OutputWeight::zero_fp32(config.vocab_size, h),
            rope,
            kv_cache,
            dominant_quant_type: oxibonsai_core::GgufTensorType::Q1_0_g128,
            has_hadamard: false,
            scratch: ModelScratch::default(),
            max_context,
            host_kv_written: 0,
            gpu_path_active: std::sync::atomic::AtomicBool::new(false),
            prefill_chunk_tokens: crate::chunked_prefill::DEFAULT_PREFILL_CHUNK_TOKENS,
            lm_head_kernel: kernel_arc,
            force_cpu_decode_after: force_cpu_decode_after_from_env(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            gpu_weight_cache: std::sync::Mutex::new(None),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            metal_q1_slots: super::forward_metal::Q1MetalSlots::fresh(),
            #[cfg(all(
                feature = "native-cuda",
                any(target_os = "linux", target_os = "windows")
            ))]
            cuda_qkv_cache: std::sync::Mutex::new(None),
            #[cfg(all(
                feature = "native-cuda",
                any(target_os = "linux", target_os = "windows")
            ))]
            cuda_model_epoch:
                oxibonsai_kernels::gpu_backend::cuda_graph_slot::next_cuda_model_epoch(),
            config,
        }
    }
}
