//! `BonsaiModel::new_for_testing_with_blocks`: a tiny, deterministic model
//! with real Transformer blocks for the prefix-cache tests. Split out of
//! `model/types/mod.rs`.
//!
//! Also the test-only [`Q1ReplicaFixture`]: a tiny all-`Q1_0_g128` model —
//! blocks **and** LM head — whose weights are leaked once and shared by every
//! replica built from them, i.e. the in-process image of engine-pool replicas
//! of one GGUF mapping.

use super::constructors::{
    effective_context, force_cpu_decode_after_from_env, prealloc_context, ModelGpuSlots,
};
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

        let output_weight = OutputWeight::zero_fp32(config.vocab_size, h);
        // MET-02: the leaked blocks anchor this model's own GPU slot namespace.
        #[cfg_attr(
            not(any(
                all(feature = "metal", target_os = "macos"),
                all(
                    feature = "native-cuda",
                    any(target_os = "linux", target_os = "windows")
                )
            )),
            allow(unused_variables)
        )]
        let gpu_slots = ModelGpuSlots::attach(&mut blocks, &output_weight);

        Self {
            // The fixture's embedding is a constant 0.01 in every element, as
            // before — now synthesized per row instead of materialized.
            token_embd: EmbeddingTable::constant(0.01, config.vocab_size, h),
            shared_embd: std::sync::Arc::from(Vec::new()),
            blocks,
            output_norm: RmsNorm::new(vec![1.0; h], config.rms_norm_eps),
            output_weight,
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
            metal_q1_slots: gpu_slots.metal_q1_slots,
            #[cfg(all(
                feature = "native-cuda",
                any(target_os = "linux", target_os = "windows")
            ))]
            cuda_qkv_cache: std::sync::Mutex::new(None),
            #[cfg(all(
                feature = "native-cuda",
                any(target_os = "linux", target_os = "windows")
            ))]
            cuda_model_epoch: gpu_slots.cuda_model_epoch,
            config,
        }
    }
}

/// One layer's leaked `Q1_0_g128` projections, in `TransformerBlock::new`
/// order: q, k, v, attention output, gate, up, down.
#[cfg(all(test, feature = "metal", target_os = "macos"))]
type Q1FixtureLayer = [&'static [oxibonsai_core::tensor::BlockQ1_0G128]; 7];

/// A tiny all-`Q1_0_g128` model — blocks **and** LM head — whose weights are
/// leaked once and shared by every [`Self::replica`], so two replicas borrow
/// the very same weight addresses exactly like two engine-pool replicas of
/// one GGUF mapping do. The layers are built on the caller's dispatcher (a
/// GPU tier gives every projection a GPU upload handle, which the Q1 fused
/// Metal paths require), and the weights are varied so every projection and
/// the LM head are genuinely different matrices. Only the Metal tests (the
/// namespace-sharing acceptance) use it, so only that build compiles it.
#[cfg(all(test, feature = "metal", target_os = "macos"))]
pub(crate) struct Q1ReplicaFixture {
    config: Qwen3Config,
    kernel: std::sync::Arc<oxibonsai_kernels::KernelDispatcher>,
    layers: Vec<Q1FixtureLayer>,
    norms: &'static [f32],
    lm_head: &'static [oxibonsai_core::tensor::BlockQ1_0G128],
    embedding: std::sync::Arc<[f32]>,
}

#[cfg(all(test, feature = "metal", target_os = "macos"))]
impl Q1ReplicaFixture {
    /// Leak one deterministic weight set for `config` (which must have
    /// `hidden_size` and `intermediate_size` multiples of 128).
    pub(crate) fn new(
        config: Qwen3Config,
        kernel: std::sync::Arc<oxibonsai_kernels::KernelDispatcher>,
        seed: u64,
    ) -> Self {
        let (h, inter) = (config.hidden_size, config.intermediate_size);
        let (nq, nkv, hd) = (
            config.num_attention_heads,
            config.num_kv_heads,
            config.head_dim,
        );
        assert!(
            h.is_multiple_of(128) && inter.is_multiple_of(128) && (nq * hd).is_multiple_of(128),
            "the Q1 replica fixture needs 128-multiple projection widths"
        );
        let layers = (0..config.num_layers)
            .map(|layer| {
                let s = seed.wrapping_add((layer as u64) << 8);
                [
                    leak_q1_blocks(nq * hd * (h / 128), s + 1),
                    leak_q1_blocks(nkv * hd * (h / 128), s + 2),
                    leak_q1_blocks(nkv * hd * (h / 128), s + 3),
                    leak_q1_blocks(h * (nq * hd / 128), s + 4),
                    leak_q1_blocks(inter * (h / 128), s + 5),
                    leak_q1_blocks(inter * (h / 128), s + 6),
                    leak_q1_blocks(h * (inter / 128), s + 7),
                ]
            })
            .collect();
        let norms: Vec<f32> = (0..h.max(hd))
            .map(|i| 0.75 + 0.5 * (((i as u64 ^ seed) % 17) as f32) / 17.0)
            .collect();
        let embedding: Vec<f32> = (0..config.vocab_size * h)
            .map(|i| 0.5 * (1.0 + 0.25 * ((i as f32) * 0.013 + seed as f32).sin()))
            .collect();
        Self {
            lm_head: leak_q1_blocks(config.vocab_size * (h / 128), seed ^ 0xABCD),
            kernel,
            layers,
            norms: Box::leak(norms.into_boxed_slice()),
            embedding: std::sync::Arc::from(embedding),
            config,
        }
    }

    /// The fixture's configuration.
    pub(crate) fn config(&self) -> &Qwen3Config {
        &self.config
    }

    /// One more replica over the shared weights: its own KV cache, scratch
    /// and GPU weight-cache marker; the weights (and therefore the GGUF-mapping
    /// namespace it joins) are the fixture's.
    pub(crate) fn replica(&self) -> crate::error::ModelResult<BonsaiModel<'static>> {
        use crate::block::TransformerBlock;
        use crate::layers::linear::{Linear1Bit, LinearLayer};
        use oxibonsai_core::tensor::BlockQ1_0G128;

        let config = self.config.clone();
        let (h, inter) = (config.hidden_size, config.intermediate_size);
        let (nq, nkv, hd) = (
            config.num_attention_heads,
            config.num_kv_heads,
            config.head_dim,
        );
        let eps = config.rms_norm_eps;
        let norm = |len: usize| RmsNorm::new(self.norms[..len].to_vec(), eps);
        let linear = |blocks: &'static [BlockQ1_0G128],
                      out: usize,
                      inp: usize|
         -> crate::error::ModelResult<LinearLayer<'static>> {
            Ok(Linear1Bit::new(blocks, out, inp, std::sync::Arc::clone(&self.kernel))?.into())
        };
        let mut blocks = Vec::with_capacity(self.layers.len());
        for (layer_idx, &[q, k, v, o, gate, up, down]) in self.layers.iter().enumerate() {
            blocks.push(TransformerBlock::new(
                layer_idx,
                norm(h),
                linear(q, nq * hd, h)?,
                linear(k, nkv * hd, h)?,
                linear(v, nkv * hd, h)?,
                linear(o, h, nq * hd)?,
                norm(hd),
                norm(hd),
                norm(h),
                linear(gate, inter, h)?,
                linear(up, inter, h)?,
                linear(down, h, inter)?,
                nq,
                nkv,
                hd,
                h,
            ));
        }
        let output_weight = OutputWeight::OneBit(Linear1Bit::new(
            self.lm_head,
            config.vocab_size,
            h,
            std::sync::Arc::clone(&self.kernel),
        )?);
        let max_context = effective_context(&config, None, None);
        let prealloc = prealloc_context(max_context, None);
        let kv_cache = KvCache::new_lazy_f16(
            config.num_layers,
            config.num_kv_heads,
            config.head_dim,
            max_context,
            prealloc.min(GROWTH_CHUNK_POSITIONS),
        );
        let rope = build_rope_table_or_unscaled(&config, prealloc);
        #[cfg_attr(
            not(any(
                all(feature = "metal", target_os = "macos"),
                all(
                    feature = "native-cuda",
                    any(target_os = "linux", target_os = "windows")
                )
            )),
            allow(unused_variables)
        )]
        let gpu_slots = ModelGpuSlots::attach(&mut blocks, &output_weight);
        Ok(BonsaiModel {
            token_embd: EmbeddingTable::dense(
                std::sync::Arc::clone(&self.embedding),
                config.vocab_size,
                h,
            ),
            shared_embd: std::sync::Arc::from(Vec::new()),
            blocks,
            output_norm: norm(h),
            output_weight,
            rope,
            kv_cache,
            dominant_quant_type: oxibonsai_core::GgufTensorType::Q1_0_g128,
            has_hadamard: false,
            scratch: ModelScratch::default(),
            max_context,
            host_kv_written: 0,
            gpu_path_active: std::sync::atomic::AtomicBool::new(false),
            prefill_chunk_tokens: crate::chunked_prefill::DEFAULT_PREFILL_CHUNK_TOKENS,
            lm_head_kernel: std::sync::Arc::clone(&self.kernel),
            force_cpu_decode_after: force_cpu_decode_after_from_env(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            gpu_weight_cache: std::sync::Mutex::new(None),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            metal_q1_slots: gpu_slots.metal_q1_slots,
            #[cfg(all(
                feature = "native-cuda",
                any(target_os = "linux", target_os = "windows")
            ))]
            cuda_qkv_cache: std::sync::Mutex::new(None),
            #[cfg(all(
                feature = "native-cuda",
                any(target_os = "linux", target_os = "windows")
            ))]
            cuda_model_epoch: gpu_slots.cuda_model_epoch,
            config,
        })
    }
}

/// `n` deterministic, varied `Q1_0_g128` blocks, leaked for `'static`.
#[cfg(all(test, feature = "metal", target_os = "macos"))]
fn leak_q1_blocks(n: usize, seed: u64) -> &'static [oxibonsai_core::tensor::BlockQ1_0G128] {
    use half::f16;
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        state
    };
    let blocks: Vec<oxibonsai_core::tensor::BlockQ1_0G128> = (0..n)
        .map(|_| {
            let scale = 0.01 + ((next() >> 40) % 1000) as f32 * 0.00002;
            let mut qs = [0u8; 16];
            for byte in qs.iter_mut() {
                *byte = (next() >> 33) as u8;
            }
            oxibonsai_core::tensor::BlockQ1_0G128 {
                d: f16::from_f32(scale),
                qs,
            }
        })
        .collect();
    Box::leak(blocks.into_boxed_slice())
}
