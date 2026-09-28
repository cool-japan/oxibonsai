//! Batched CPU prefill (perf-M2), shared by the generation prefill and the
//! embedding hidden-state pass (`RT-08`).
//!
//! Before this module there was **no batched CPU prefill at all**:
//! [`BonsaiModel::forward_prefill`](super::BonsaiModel::forward_prefill)
//! tried the fused Metal/CUDA batch paths and then fell through to
//! `forward_sequential`, a plain `for token { self.forward(...) }` loop. On a
//! CPU-only build that meant
//!
//! * every projection ran as a **GEMV** — one pass over the whole quantized
//!   weight matrix per prompt token, decoded from scratch each time (K-18);
//!   and
//! * the **LM head ran at every prompt position** and its `[vocab]` logits
//!   were thrown away for all but the last — for Bonsai 2 that is
//!   `248 320 x 5120` MACs discarded per prompt token.
//!
//! The consequence was measured at **183 ms/prompt-token** (277 prompt
//! tokens in 50.81 s on the default build), *worse* than the 125 ms CPU
//! decode step, which at least has the excuse of being memory-bound.
//!
//! # Shape of one pass
//!
//! [`BonsaiModel::forward_prefill_cpu`] and the embedding pass
//! ([`BonsaiModel::forward_hidden`]) both drive `run_prefill_cpu`:
//!
//! 1. the prompt is embedded once per micro-batch;
//! 2. every projection (Q/K/V, attention output, FFN gate/up/down) is a
//!    **GEMM over the whole micro-batch** through the K-18 register-blocked
//!    kernels (`gemm_tq2_0_g128_blocked` / `gemm_1bit_g128_blocked`), which
//!    decode each weight block once per `MR` batch rows — split into Rayon
//!    tasks over batch slabs **and** feature slabs (see "GEMM split");
//! 3. QK-norm, RoPE and the KV-cache writes run **strictly in position
//!    order**; the causal GQA attention of the micro-batch's rows then runs
//!    in parallel over rows, each row reading exactly positions `0..=pos`
//!    (see "Attention");
//! 4. each micro-batch's post-block rows go to a sink together with
//!    `output_norm`: the generation prefill keeps the last row and runs the
//!    output norm and the **LM head once per call**; the embedding pass
//!    normalises every row and never touches the head.
//!
//! Long prompts are processed in [`CPU_PREFILL_MICRO_BATCH`]-token passes so
//! the activation buffers stay bounded (`m x intermediate_size` floats is
//! 35 MB per buffer at `m = 512` for Bonsai 2). Each pass attends over every
//! position committed by the passes before it, so the result is the same as
//! one big pass would give — and the decode count is unchanged, because the
//! register-blocked GEMM decodes each weight block once per `MR` rows
//! however the batch is split. This micro-batching is internal to one call;
//! it is not the same split as
//! [`crate::chunked_prefill::run_chunked_prefill`]'s outer chunks. When a
//! long prompt is chunked there, `forward_prefill_unchunked`
//! — and therefore this function — runs once per outer chunk, so the LM
//! head runs once *per chunk*, not once per prompt; only the last chunk's
//! logits reach the caller. That repeat is bounded by the chunk count, not
//! the prompt length, so it keeps perf-M2's win (was: once per **token**).
//!
//! # GEMM split (bit-identical)
//!
//! `oxibonsai_kernels::parallel::gemm_*_par` splits Rayon work over batch
//! slabs only, one slab per worker and at least one register block (8 rows)
//! each. That fills the pool for a 128-row micro-batch, but a short input
//! gets `ceil(m / 8)` slabs: a 10-token prompt ran every GEMM on **two
//! cores** of eight. `gemm_blocked_2d` keeps each batch slab to one register
//! block and adds feature slabs until there are about four tasks per worker
//! thread, so a short input uses the whole machine and work stealing
//! balances the long tail on a mixed performance/efficiency-core part. Every
//! task runs the same register-blocked kernel on the same dispatcher, whose
//! per-element accumulation order depends on neither which batch rows nor
//! which features share a call, so the output is **bit-identical** to
//! `gemm_*_par` (pinned by `gemm_blocked_2d_is_bit_identical_to_the_kernel_driver`
//! over both formats and every batch size a micro-batch can take).
//!
//! The opt-in INT8 dot-product tier (`OXIBONSAI_KERNEL_TIER`) is routed
//! inside the kernel crate's drivers, so while it is selected the GEMMs go
//! through `gemm_*_par` unchanged; with the variable unset that route is
//! never taken.
//!
//! # Attention (bit-identical)
//!
//! Row `i` of a micro-batch must see exactly the keys and values of
//! positions `0..=pos_start + i`. The rows' keys and values are stored in
//! position order first; the rows' attention then runs in parallel, each
//! through the very function the per-token decode path runs
//! (`gqa_attention`, `seq_len = pos + 1`), and the cache's `attend_group`
//! reads exactly the first `seq_len` stored positions. A later row's keys
//! being in the cache already is therefore invisible to an earlier row, and
//! each row gets the bits it would get alone (pinned by
//! `attend_rows_is_bit_identical_to_the_per_row_attention`, `f32` and
//! `f16` caches). Every position of the call is made resident before the
//! first layer runs, so the cache's allocation cannot move in between.
//!
//! # MEASUREMENTS
//!
//! `real_model_cpu_prefill_outruns_the_sequential_prefill` runs
//! [`BonsaiModel::forward_prefill_cpu`] against the sequential per-token
//! reference (same pinned CPU tier on both sides) on the real, shipped
//! `Ternary-Bonsai-1.7B.gguf`, 280 prompt tokens, release, `--all-features`,
//! on an 8-core M3. With the batch-slab split, three independent measurement
//! points (different checkouts, different days, different machine load)
//! gave:
//!
//! | # | runs | batched (ms/prompt-token) | sequential | speedup | load avg (1/5/15) |
//! |---|---|---|---|---|---|
//! | 1 | single run | **82.3** | 268.6 | 3.26x | 14.87 / 34.43 / 42.60 |
//! | 2 | single run | **82.707** | 392.4 | 4.74x | 93.85 |
//! | 3 | min of 3 in-process runs, two sessions | **25.9** and **31.0** (per-run 34.748 / 31.488 / 30.989) | 279.6 and 305.1 | 10.78x / 9.84x | 13.19-15.60 before, 18.75-20.22 after |
//!
//! `cos(batched, sequential)` printed **0.9999999999997823** — identical to
//! all thirteen digits — in every one of those runs, which is how each is
//! known to have timed the same code path. The correctness leg (cos >=
//! 0.9999) and the relative leg (speedup >= 3x) hold across a 7x
//! load-average range; those dimensionless figures are what this path's
//! acceptance rests on. The spread of the absolute figure (82 against 26-31
//! ms/token) was never isolated beyond this: not file I/O or model load
//! (both finish before the timer starts), not in-process warm-up alone (point
//! 3's first run is already 2.4x faster than 82.3), not a code change on the
//! GEMM path; what remains is the load average and single-run against
//! min-of-three — at load 93.85 on 8 cores, a Rayon-parallel GEMM running
//! 2-3x slow is an ordinary outcome. The absolute figure is therefore
//! recorded, not asserted.
//!
//! The two-way GEMM split and the row-parallel attention keep every output
//! bit-identical — pinned by `gemm_blocked_2d_is_bit_identical_to_the_kernel_driver`,
//! `both_gemm_routes_agree_bit_for_bit` and
//! `attend_rows_is_bit_identical_to_the_per_row_attention`, and confirmed
//! end to end against the batch-slab version's logits on the real 1.7B (10-
//! and 280-token prompts, every bit equal). Interleaved on this M3, the split
//! runs the 10-row projection GEMMs 1.3-2.1x faster than the batch-slab
//! driver and the 128-row ones at parity; the embedding path's end-to-end
//! effect is measured by the runtime's `embed_bench_short_and_long`, whose
//! printed figures carry their own load averages.
//!
//! # CROSS-TIER NUMERICS (real `Ternary-Bonsai-1.7B.gguf`)
//!
//! Measured over 64 self-generated greedy steps on all three of the legacy
//! parity gate's prompt slots:
//!
//! * `KernelTier::Reference` vs the auto-detected CPU tier (NEON here),
//!   batched prefill: **exactly 0.0 at every step**, identical token chains;
//! * the same pair with a per-token (`forward`) prefill: **also exactly
//!   0.0**;
//! * batched prefill vs per-token prefill *within* a tier: worst
//!   **9.06e-6 at step 45** (slot 0) and **8.58e-6 at step 0** (slot 1),
//!   token chains identical, and the same series on both tiers to every
//!   printed digit.
//!
//! So this path costs ~1e-5 absolute against the per-token sweep and
//! contributes **zero** cross-tier divergence — it is not the source of any
//! Reference-vs-Metal divergence, which is a bound-shape problem. Both
//! halves are asserted rather than merely reported:
//! `model::types::tests::forward_prefill_is_bit_exact_across_the_reference_and_native_cpu_tiers`
//! and
//! `model::types::tests::batched_prefill_agrees_with_the_per_token_prefill_within_a_tier`.
//!
//! **Scope.** This is the CPU path and only the CPU path. It handles the two
//! weight formats that have register-blocked GEMM kernels — `Q1_0_g128` and
//! `TQ2_0_g128`. Anything else (FP8, Q4\_0/Q8\_0, the K-quants, a
//! mixed-format layer, a geometry that does not match the config, a
//! declared sliding window — this path's attention is full-causal, M-17)
//! returns `Ok(None)` **before writing anything**, so the caller keeps its
//! existing, proven sequential behaviour. The LM head is not restricted: it
//! runs once through `apply_lm_head`, whatever its format.

use oxibonsai_core::tensor::{BlockQ1_0G128, QK1_0_G128};
use oxibonsai_core::BlockTQ2_0_g128;
use oxibonsai_kernels::error::KernelError;
use oxibonsai_kernels::KernelDispatcher;
use rayon::prelude::*;
use std::sync::OnceLock;

use crate::block::TransformerBlock;
use crate::error::{ModelError, ModelResult};
use crate::kv_cache::KvCache;
use crate::layers::rms_norm::RmsNorm;
use crate::layers::rope::RopeTable;
use crate::layers::swiglu::try_swiglu;

use super::BonsaiModel;

/// Shortest prompt worth batching.
///
/// A single token *is* the decode path; two tokens already amortize one
/// weight-matrix decode over two rows.
pub const CPU_PREFILL_MIN_TOKENS: usize = 2;

/// Prompt positions processed per batched pass.
///
/// Bounds the activation working set — the FFN buffers are
/// `m x intermediate_size` floats each, 8.9 MB apiece at this value for
/// Bonsai 2's `17408` — without costing anything: the register-blocked GEMM
/// decodes each weight block once per `MR` rows regardless of how the batch
/// is split, so `ceil(m / MR)` decodes happen either way.
pub const CPU_PREFILL_MICRO_BATCH: usize = 128;

/// Both register-blocked formats group this many weights per block.
const GROUP_WEIGHTS: usize = QK1_0_G128;

/// Rayon tasks a batched GEMM is split into, per worker thread.
///
/// Several tasks per thread rather than one: this machine class mixes
/// performance and efficiency cores, and a split into exactly one task per
/// thread finishes when the slowest core does. Four lets work stealing move
/// the tail onto whichever cores are free.
const GEMM_TASKS_PER_THREAD: usize = 4;

/// Fewest output features one GEMM task computes.
///
/// Below this, a task's fixed costs (the kernel's shape validation, one
/// small allocation, the Rayon hand-off) stop being negligible against its
/// arithmetic.
const GEMM_MIN_FEATURES_PER_TASK: usize = 32;

/// The dispatcher the batched CPU prefill runs its GEMMs on.
///
/// Pinned to the best **CPU** tier
/// ([`oxibonsai_kernels::cpu_kernel_tier`]) and built once per process,
/// rather than `auto_detect`. Two callers reach this path and neither wants
/// its GEMMs back on a GPU: the generation prefill only runs here once a GPU
/// prefill was declined — because there is no GPU, because the caller forced
/// the CPU (`OXIBONSAI_FORCE_CPU_DECODE_AFTER`), or because the fused GPU
/// prefill failed — so routing the GEMMs back onto the GPU would defeat
/// every one of those reasons and would put device work behind a host-KV
/// contract (MET-05); and
/// [`forward_hidden`](BonsaiModel::forward_hidden) (embeddings) runs here on
/// *every* tier, including a GPU engine's, because no head-free batched GPU
/// prefill exists (see that module's docs).
fn prefill_dispatcher() -> &'static KernelDispatcher {
    static DISPATCHER: OnceLock<KernelDispatcher> = OnceLock::new();
    DISPATCHER.get_or_init(|| KernelDispatcher::with_tier(oxibonsai_kernels::cpu_kernel_tier()))
}

/// Which driver a batched pass sends its GEMMs through, decided once per
/// call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GemmRoute {
    /// The f32 register-blocked kernels under this module's two-way split
    /// ([`gemm_blocked_2d`]) — the default.
    Blocked2d,
    /// `oxibonsai_kernels::parallel::gemm_*_par`, taken while the opt-in
    /// INT8 dot-product tier is selected through `OXIBONSAI_KERNEL_TIER`:
    /// that tier is routed inside the kernel crate's drivers, so deferring
    /// to them is what keeps the opt-in reachable from this path. With the
    /// variable unset this route is never taken.
    KernelDriver,
}

impl GemmRoute {
    /// The route for one call, read from the environment once.
    fn for_this_call() -> Self {
        if oxibonsai_kernels::dispatch_int8::Int8Tier::from_env().is_some() {
            Self::KernelDriver
        } else {
            Self::Blocked2d
        }
    }
}

/// One quantized projection matrix in a format that has a register-blocked
/// GEMM kernel.
#[derive(Clone, Copy)]
enum PrefillMatrix<'w> {
    /// `Q1_0_g128` weights.
    OneBit(&'w [BlockQ1_0G128]),
    /// `TQ2_0_g128` (ternary) weights.
    Ternary(&'w [BlockTQ2_0_g128]),
}

impl PrefillMatrix<'_> {
    /// Number of 128-weight groups this matrix holds.
    fn block_count(&self) -> usize {
        match self {
            Self::OneBit(b) => b.len(),
            Self::Ternary(b) => b.len(),
        }
    }

    /// Batch rows the format's register-blocked kernel consumes one decoded
    /// weight block with.
    fn register_block(&self) -> usize {
        match self {
            Self::OneBit(_) => oxibonsai_kernels::gemm_onebit::ONEBIT_GEMM_MR,
            Self::Ternary(_) => oxibonsai_kernels::gemm_ternary::TERNARY_GEMM_MR,
        }
    }

    /// Output rows implied by the block count and `in_features`.
    ///
    /// Derived from the weights themselves rather than assumed from the
    /// config, so a projection whose shape does not divide cleanly is
    /// declined here instead of mis-indexed later.
    fn out_features(&self, in_features: usize) -> Option<usize> {
        if in_features == 0 || !in_features.is_multiple_of(GROUP_WEIGHTS) {
            return None;
        }
        let blocks_per_row = in_features / GROUP_WEIGHTS;
        let total = self.block_count();
        if total == 0 || !total.is_multiple_of(blocks_per_row) {
            return None;
        }
        Some(total / blocks_per_row)
    }

    /// `output[m x n_rows] = input[m x k] . weights^T` on the route chosen
    /// for this call.
    #[allow(
        clippy::too_many_arguments,
        reason = "one GEMM: its route, both buffers and the three dimensions"
    )]
    fn gemm(
        &self,
        route: GemmRoute,
        input: &[f32],
        output: &mut [f32],
        m: usize,
        n_rows: usize,
        k: usize,
    ) -> ModelResult<()> {
        let dispatcher = prefill_dispatcher();
        match route {
            GemmRoute::Blocked2d => gemm_blocked_2d(*self, dispatcher, input, output, m, n_rows, k),
            GemmRoute::KernelDriver => match self {
                Self::OneBit(blocks) => oxibonsai_kernels::parallel::gemm_1bit_g128_par(
                    dispatcher, blocks, input, output, m, n_rows, k,
                ),
                Self::Ternary(blocks) => oxibonsai_kernels::parallel::gemm_ternary_g128_par(
                    dispatcher, blocks, input, output, m, n_rows, k,
                ),
            }
            .map_err(ModelError::Kernel),
        }
    }

    /// The format's register-blocked kernel over the weight rows whose
    /// blocks are `blocks`, writing `[m x n_rows]` contiguously.
    #[allow(
        clippy::too_many_arguments,
        reason = "one kernel call: its dispatcher, block range, both buffers and three dimensions"
    )]
    fn blocked_gemm(
        &self,
        dispatcher: &KernelDispatcher,
        blocks: std::ops::Range<usize>,
        input: &[f32],
        output: &mut [f32],
        m: usize,
        n_rows: usize,
        k: usize,
    ) -> ModelResult<()> {
        let available = self.block_count();
        let out_of_range = || {
            ModelError::Kernel(KernelError::buffer_too_small(
                "blocks", blocks.end, available,
            ))
        };
        match self {
            Self::OneBit(all) => oxibonsai_kernels::gemm_onebit::gemm_1bit_g128_blocked(
                dispatcher,
                all.get(blocks.clone()).ok_or_else(out_of_range)?,
                input,
                output,
                m,
                n_rows,
                k,
            ),
            Self::Ternary(all) => oxibonsai_kernels::gemm_ternary::gemm_tq2_0_g128_blocked(
                dispatcher,
                all.get(blocks.clone()).ok_or_else(out_of_range)?,
                input,
                output,
                m,
                n_rows,
                k,
            ),
        }
        .map_err(ModelError::Kernel)
    }
}

/// How [`gemm_blocked_2d`] cuts one `[m x n_rows]` GEMM into Rayon tasks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GemmSplit {
    /// Batch rows per task: one register block (the last slab may be
    /// shorter).
    row_slab: usize,
    /// Output features per task (the last slab may be narrower).
    features_per_slab: usize,
    /// Feature slabs per batch slab; every one is non-empty.
    feature_slabs: usize,
}

impl GemmSplit {
    /// Plan the split of an `[m x n_rows]` GEMM whose kernel register-blocks
    /// `mr` batch rows, for `threads` Rayon workers.
    ///
    /// Batch slabs are exactly one register block, so no slab strands a
    /// partial block that a wider slab would have filled: the weight matrix
    /// is decoded `ceil(m / mr)` times in total, the same as any split whose
    /// slabs are whole register blocks. The feature split then supplies the
    /// parallelism the batch split alone cannot — at `m = 10` a batch-only
    /// split yields two tasks, and so two busy cores, whatever the machine.
    fn plan(m: usize, n_rows: usize, mr: usize, threads: usize) -> Self {
        let row_slab = mr.max(1).min(m.max(1));
        let row_slabs = m.max(1).div_ceil(row_slab);
        let target_tasks = threads.max(1).saturating_mul(GEMM_TASKS_PER_THREAD);
        let wanted = target_tasks.div_ceil(row_slabs).max(1);
        let most = n_rows.div_ceil(GEMM_MIN_FEATURES_PER_TASK).max(1);
        let features_per_slab = n_rows.max(1).div_ceil(wanted.min(most));
        let feature_slabs = n_rows.max(1).div_ceil(features_per_slab);
        Self {
            row_slab,
            features_per_slab,
            feature_slabs,
        }
    }
}

/// `output[m x n_rows] = input[m x k] . weights^T` through the format's
/// register-blocked kernel, split over batch slabs **and** feature slabs.
///
/// **Bit-identical to `oxibonsai_kernels::parallel::gemm_*_par`**: both run
/// the same `gemm_*_blocked` kernel on the same dispatcher, and that kernel
/// computes every `(batch row, feature)` element with an accumulation order
/// that depends on neither which other rows nor which other features share
/// the call (the kernel's own module documents that per-element contract).
/// Only the task boundaries differ — which `gemm_blocked_2d_is_bit_identical_to_the_kernel_driver`
/// pins over odd shapes, both formats and every batch size a micro-batch
/// can take.
///
/// Each task computes its `[rows x features]` tile into a private buffer;
/// the batch slab that owns those output rows then scatters its tiles into
/// place, so no two tasks ever write the same memory.
fn gemm_blocked_2d(
    matrix: PrefillMatrix<'_>,
    dispatcher: &KernelDispatcher,
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    k: usize,
) -> ModelResult<()> {
    if m == 0 || n_rows == 0 {
        return Ok(());
    }
    if k == 0 || !k.is_multiple_of(GROUP_WEIGHTS) {
        return Err(ModelError::Kernel(KernelError::NotBlockAligned {
            count: k,
            block_size: GROUP_WEIGHTS,
        }));
    }
    let input_len = m
        .checked_mul(k)
        .ok_or_else(|| gemm_overflow("input", m, k))?;
    let output_len = m
        .checked_mul(n_rows)
        .ok_or_else(|| gemm_overflow("output", m, n_rows))?;
    let supplied = input.len();
    let input = input.get(..input_len).ok_or_else(|| {
        ModelError::Kernel(KernelError::dimension_mismatch(
            "input", input_len, supplied,
        ))
    })?;
    let available = output.len();
    let output = output.get_mut(..output_len).ok_or_else(|| {
        ModelError::Kernel(KernelError::buffer_too_small(
            "output", output_len, available,
        ))
    })?;
    let blocks_per_row = k / GROUP_WEIGHTS;
    let needed_blocks = n_rows
        .checked_mul(blocks_per_row)
        .ok_or_else(|| gemm_overflow("blocks", n_rows, blocks_per_row))?;
    if matrix.block_count() < needed_blocks {
        return Err(ModelError::Kernel(KernelError::buffer_too_small(
            "blocks",
            needed_blocks,
            matrix.block_count(),
        )));
    }

    let split = GemmSplit::plan(
        m,
        n_rows,
        matrix.register_block(),
        rayon::current_num_threads(),
    );
    output
        .par_chunks_mut(split.row_slab * n_rows)
        .enumerate()
        .try_for_each(|(slab, out_slab)| -> ModelResult<()> {
            let rows = out_slab.len() / n_rows;
            let first = slab * split.row_slab;
            let input_slab = input
                .get(first * k..(first + rows) * k)
                .ok_or_else(|| gemm_overflow("input slab", first, rows))?;
            let tiles = (0..split.feature_slabs)
                .into_par_iter()
                .map(|feature_slab| -> ModelResult<(usize, Vec<f32>)> {
                    let start = feature_slab * split.features_per_slab;
                    let end = (start + split.features_per_slab).min(n_rows);
                    let width = end - start;
                    let mut tile = vec![0.0f32; rows * width];
                    matrix.blocked_gemm(
                        dispatcher,
                        start * blocks_per_row..end * blocks_per_row,
                        input_slab,
                        &mut tile,
                        rows,
                        width,
                        k,
                    )?;
                    Ok((start, tile))
                })
                .collect::<ModelResult<Vec<_>>>()?;
            for (start, tile) in tiles {
                let width = tile.len() / rows;
                for (dst_row, src_row) in out_slab
                    .chunks_exact_mut(n_rows)
                    .zip(tile.chunks_exact(width))
                {
                    dst_row[start..start + width].copy_from_slice(src_row);
                }
            }
            Ok(())
        })
}

/// A GEMM dimension product that does not fit `usize` — unreachable for any
/// shape that fits in memory, reported rather than wrapped.
fn gemm_overflow(what: &str, a: usize, b: usize) -> ModelError {
    ModelError::ShapeInvariant {
        tensor: format!("batched prefill GEMM {what}"),
        expected: "dimensions whose product fits usize".to_string(),
        actual: format!("{a} x {b}"),
    }
}

/// Everything one transformer layer contributes to a batched pass.
///
/// All four RMSNorms of a block are constructed with `config.rms_norm_eps`
/// (see `weight_loaders.rs`), and only `attn_norm`'s epsilon is exposed by
/// [`TransformerBlock`]'s public accessors, so [`Self::eps`] is that one
/// value and is used for all four — exactly the number the per-token path
/// uses.
struct LayerPlan<'w> {
    attn_q: PrefillMatrix<'w>,
    attn_k: PrefillMatrix<'w>,
    attn_v: PrefillMatrix<'w>,
    attn_output: PrefillMatrix<'w>,
    ffn_gate: PrefillMatrix<'w>,
    ffn_up: PrefillMatrix<'w>,
    ffn_down: PrefillMatrix<'w>,
    attn_norm_w: &'w [f32],
    q_norm_w: &'w [f32],
    k_norm_w: &'w [f32],
    ffn_norm_w: &'w [f32],
    eps: f32,
}

impl LayerPlan<'_> {
    /// Does every projection's derived shape agree with the model geometry?
    fn shapes_match(&self, geom: &PrefillGeometry) -> bool {
        self.attn_q.out_features(geom.hidden) == Some(geom.q_dim)
            && self.attn_k.out_features(geom.hidden) == Some(geom.kv_dim)
            && self.attn_v.out_features(geom.hidden) == Some(geom.kv_dim)
            && self.attn_output.out_features(geom.q_dim) == Some(geom.hidden)
            && self.ffn_gate.out_features(geom.hidden) == Some(geom.intermediate)
            && self.ffn_up.out_features(geom.hidden) == Some(geom.intermediate)
            && self.ffn_down.out_features(geom.intermediate) == Some(geom.hidden)
            && self.attn_norm_w.len() == geom.hidden
            && self.ffn_norm_w.len() == geom.hidden
            && self.q_norm_w.len() == geom.head_dim
            && self.k_norm_w.len() == geom.head_dim
    }
}

/// Collect a layer's seven projections if — and only if — all seven are in
/// the *same* register-blocked format.
///
/// A mixed-format layer returns `None` rather than being partially batched:
/// the caller then keeps the sequential path for the whole prompt, which is
/// always correct.
fn layer_plan<'w>(block: &'w TransformerBlock<'_>) -> Option<LayerPlan<'w>> {
    let norms = (
        block.attn_norm_weight(),
        block.q_norm_weight(),
        block.k_norm_weight(),
        block.ffn_norm_weight(),
        block.attn_norm_eps(),
    );
    if let (Some(q), Some(k), Some(v), Some(o), Some(g), Some(u), Some(d)) = (
        block.attn_q_blocks(),
        block.attn_k_blocks(),
        block.attn_v_blocks(),
        block.attn_output_blocks(),
        block.ffn_gate_blocks(),
        block.ffn_up_blocks(),
        block.ffn_down_blocks(),
    ) {
        return Some(LayerPlan {
            attn_q: PrefillMatrix::OneBit(q),
            attn_k: PrefillMatrix::OneBit(k),
            attn_v: PrefillMatrix::OneBit(v),
            attn_output: PrefillMatrix::OneBit(o),
            ffn_gate: PrefillMatrix::OneBit(g),
            ffn_up: PrefillMatrix::OneBit(u),
            ffn_down: PrefillMatrix::OneBit(d),
            attn_norm_w: norms.0,
            q_norm_w: norms.1,
            k_norm_w: norms.2,
            ffn_norm_w: norms.3,
            eps: norms.4,
        });
    }
    if let (Some(q), Some(k), Some(v), Some(o), Some(g), Some(u), Some(d)) = (
        block.attn_q_blocks_ternary(),
        block.attn_k_blocks_ternary(),
        block.attn_v_blocks_ternary(),
        block.attn_output_blocks_ternary(),
        block.ffn_gate_blocks_ternary(),
        block.ffn_up_blocks_ternary(),
        block.ffn_down_blocks_ternary(),
    ) {
        return Some(LayerPlan {
            attn_q: PrefillMatrix::Ternary(q),
            attn_k: PrefillMatrix::Ternary(k),
            attn_v: PrefillMatrix::Ternary(v),
            attn_output: PrefillMatrix::Ternary(o),
            ffn_gate: PrefillMatrix::Ternary(g),
            ffn_up: PrefillMatrix::Ternary(u),
            ffn_down: PrefillMatrix::Ternary(d),
            attn_norm_w: norms.0,
            q_norm_w: norms.1,
            k_norm_w: norms.2,
            ffn_norm_w: norms.3,
            eps: norms.4,
        });
    }
    None
}

/// Geometry shared by every layer of a batched pass.
#[derive(Debug, Clone, Copy)]
struct PrefillGeometry {
    hidden: usize,
    intermediate: usize,
    q_dim: usize,
    kv_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    heads_per_group: usize,
}

/// Scratch buffers for one batched pass, sized once and reused per layer.
///
/// `q_rope` holds every row of the micro-batch (the attention of all rows
/// runs after their keys and values are stored); `q_normed`, `k_normed` and
/// `k_rope` are one-row staging buffers.
struct PrefillScratch {
    hidden: Vec<f32>,
    normed: Vec<f32>,
    q_all: Vec<f32>,
    k_all: Vec<f32>,
    v_all: Vec<f32>,
    q_normed: Vec<f32>,
    k_normed: Vec<f32>,
    q_rope: Vec<f32>,
    k_rope: Vec<f32>,
    attn_out: Vec<f32>,
    proj: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    swiglu: Vec<f32>,
}

impl PrefillScratch {
    fn new(batch: usize, geom: &PrefillGeometry) -> Self {
        Self {
            hidden: vec![0.0; batch * geom.hidden],
            normed: vec![0.0; batch * geom.hidden],
            q_all: vec![0.0; batch * geom.q_dim],
            k_all: vec![0.0; batch * geom.kv_dim],
            v_all: vec![0.0; batch * geom.kv_dim],
            q_normed: vec![0.0; geom.q_dim],
            k_normed: vec![0.0; geom.kv_dim],
            q_rope: vec![0.0; batch * geom.q_dim],
            k_rope: vec![0.0; geom.kv_dim],
            attn_out: vec![0.0; batch * geom.q_dim],
            proj: vec![0.0; batch * geom.hidden],
            gate: vec![0.0; batch * geom.intermediate],
            up: vec![0.0; batch * geom.intermediate],
            swiglu: vec![0.0; batch * geom.intermediate],
        }
    }
}

/// One micro-batch of post-block hidden rows, as a batched pass hands it to
/// the sink of [`BonsaiModel::run_prefill_cpu`].
#[derive(Debug, Clone, Copy)]
pub(super) struct PrefillRows<'r> {
    /// Index, within the call's `token_ids`, of the first row.
    pub(super) first: usize,
    /// Number of rows (at least one).
    pub(super) count: usize,
    /// `[count x hidden]` hidden states after the last block and **before**
    /// `output_norm`.
    pub(super) data: &'r [f32],
}

impl BonsaiModel<'_> {
    /// Batched CPU prefill of `token_ids` starting at `pos_start` (perf-M2).
    ///
    /// Returns `Ok(Some(logits))` — the `[vocab_size]` logits of the **last**
    /// prompt position, matching
    /// [`forward_prefill`](Self::forward_prefill)'s contract — after
    /// committing every prompt position's keys and values to the host KV
    /// cache.
    ///
    /// Returns `Ok(None)` when this path declines the prompt, in which case
    /// **nothing has been written**: no KV-cache entry, no watermark. The
    /// caller must fall back to its sequential path. That happens for a
    /// prompt below `CPU_PREFILL_MIN_TOKENS` (2), a model with no transformer
    /// blocks, a model that declares a sliding attention window (M-17: this
    /// path's attention is full-causal), a layer whose projections are not
    /// all in one register-blocked format (`Q1_0_g128` / `TQ2_0_g128`), or a
    /// geometry that does not match the config — every decline is decided
    /// before the first write.
    ///
    /// # Errors
    ///
    /// * [`ModelError::SequenceTooLong`] — the last prompt position is beyond
    ///   the effective context (sec-11).
    /// * The distinguished GPU-fallback error — a GPU path has been
    ///   maintaining a device KV cache for this sequence, so the host cache
    ///   this path reads is not coherent (MET-05).
    /// * [`ModelError::Kernel`] — a GEMM or a norm rejected its shapes.
    pub fn forward_prefill_cpu(
        &mut self,
        token_ids: &[u32],
        pos_start: usize,
    ) -> ModelResult<Option<Vec<f32>>> {
        let h = self.config.hidden_size;
        let total = token_ids.len();
        // perf-M2: only the last position feeds the LM head, and it is the
        // last row of the final micro-batch.
        let mut last_row = vec![0.0f32; h];
        let ran = self.run_prefill_cpu(token_ids, pos_start, |_, rows| {
            if rows.first + rows.count == total {
                let start = (rows.count - 1) * h;
                let src = rows
                    .data
                    .get(start..start + h)
                    .ok_or_else(|| prefill_rows_short(rows.count, h, rows.data.len()))?;
                last_row.copy_from_slice(src);
            }
            Ok(())
        })?;
        if !ran {
            return Ok(None);
        }
        // perf-M2: the output norm and the LM head run exactly once.
        let mut normed = vec![0.0f32; h];
        self.output_norm.forward(&last_row, &mut normed)?;
        let mut logits = vec![0.0f32; self.config.vocab_size];
        self.apply_lm_head(&normed, &mut logits)?;
        Ok(Some(logits))
    }

    /// The batched CPU pass itself: embed, run every block over
    /// [`CPU_PREFILL_MICRO_BATCH`]-row micro-batches, commit every position's
    /// keys and values to the host KV cache, and hand each micro-batch's
    /// post-block rows to `sink` together with this model's `output_norm`.
    ///
    /// Shared by [`forward_prefill_cpu`](Self::forward_prefill_cpu) (which
    /// keeps only the last row, for the LM head) and
    /// [`forward_hidden`](Self::forward_hidden) (which normalises every row),
    /// so the two cannot compute different hidden states.
    ///
    /// Returns `Ok(false)` — having written nothing — for exactly the
    /// declines [`forward_prefill_cpu`](Self::forward_prefill_cpu) lists.
    ///
    /// # Errors
    ///
    /// As [`forward_prefill_cpu`](Self::forward_prefill_cpu), plus whatever
    /// `sink` returns.
    pub(super) fn run_prefill_cpu<F>(
        &mut self,
        token_ids: &[u32],
        pos_start: usize,
        mut sink: F,
    ) -> ModelResult<bool>
    where
        F: FnMut(&RmsNorm, PrefillRows<'_>) -> ModelResult<()>,
    {
        let total = token_ids.len();
        if total < CPU_PREFILL_MIN_TOKENS
            || self.blocks.is_empty()
            || self.config.sliding_window.is_some()
        {
            return Ok(false);
        }
        let Some(geom) = self.prefill_geometry() else {
            return Ok(false);
        };
        // Decide, before touching any state, whether every layer is in
        // scope. `layer_plan` borrows `self.blocks`, so this probe is run
        // and dropped before the `&mut self` calls below.
        if !self
            .blocks
            .iter()
            .all(|b| layer_plan(b).is_some_and(|p| p.shapes_match(&geom)))
        {
            return Ok(false);
        }

        let last_pos = pos_start
            .checked_add(total - 1)
            .ok_or_else(|| ModelError::Internal("prefill position overflow".to_string()))?;
        self.ensure_context_capacity(last_pos)?;
        self.require_host_kv_coherent(pos_start)?;
        if self.kv_cache.max_seq_len() <= last_pos {
            return Ok(false);
        }

        let route = GemmRoute::for_this_call();
        let batch = CPU_PREFILL_MICRO_BATCH.min(total);
        let mut scratch = PrefillScratch::new(batch, &geom);
        let h = geom.hidden;

        {
            // Disjoint field borrows: `plans` holds `&self.blocks`, the pass
            // takes `&mut self.kv_cache` and `&self.rope`, the sink gets
            // `&self.output_norm`.
            let plans: Vec<LayerPlan<'_>> = self.blocks.iter().filter_map(layer_plan).collect();
            if plans.len() != self.blocks.len() {
                return Ok(false);
            }
            let mut offset = 0usize;
            while offset < total {
                let rows = batch.min(total - offset);
                // `copy_rows` exists only on the GPU-feature builds, so embed
                // row by row through the always-available `copy_row`.
                for (t, &token_id) in token_ids[offset..offset + rows].iter().enumerate() {
                    self.token_embd
                        .copy_row(token_id, &mut scratch.hidden[t * h..(t + 1) * h])?;
                }
                prefill_pass(
                    &plans,
                    &self.rope,
                    &mut self.kv_cache,
                    &geom,
                    &mut scratch,
                    PassRange {
                        rows,
                        pos_start: pos_start + offset,
                    },
                    route,
                )?;
                sink(
                    &self.output_norm,
                    PrefillRows {
                        first: offset,
                        count: rows,
                        data: &scratch.hidden[..rows * h],
                    },
                )?;
                offset += rows;
            }
        }
        self.note_host_kv_written(last_pos);
        Ok(true)
    }

    /// Geometry of this model, or `None` if it is not a shape this path
    /// handles.
    fn prefill_geometry(&self) -> Option<PrefillGeometry> {
        let cfg = &self.config;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        if num_heads == 0 || num_kv_heads == 0 || head_dim == 0 {
            return None;
        }
        if !num_heads.is_multiple_of(num_kv_heads) {
            return None;
        }
        if !cfg.hidden_size.is_multiple_of(GROUP_WEIGHTS)
            || !cfg.intermediate_size.is_multiple_of(GROUP_WEIGHTS)
        {
            return None;
        }
        let q_dim = num_heads.checked_mul(head_dim)?;
        let kv_dim = num_kv_heads.checked_mul(head_dim)?;
        if !q_dim.is_multiple_of(GROUP_WEIGHTS) {
            return None;
        }
        if self.kv_cache.num_layers() < self.blocks.len() || self.kv_cache.head_dim() != head_dim {
            return None;
        }
        Some(PrefillGeometry {
            hidden: cfg.hidden_size,
            intermediate: cfg.intermediate_size,
            q_dim,
            kv_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            heads_per_group: num_heads / num_kv_heads,
        })
    }
}

/// A sink got fewer floats than its row count says it holds — an internal
/// invariant violation, reported rather than indexed past.
fn prefill_rows_short(count: usize, hidden: usize, len: usize) -> ModelError {
    ModelError::ShapeInvariant {
        tensor: "batched prefill rows".to_string(),
        expected: format!("{count} rows x {hidden} floats"),
        actual: format!("{len} floats"),
    }
}

/// Which prompt positions one batched pass covers.
#[derive(Debug, Clone, Copy)]
struct PassRange {
    rows: usize,
    pos_start: usize,
}

/// One batched pass over `range.rows` already-embedded positions.
///
/// `scratch.hidden[..rows * hidden]` must hold the embeddings on entry and
/// holds the post-block hidden states on exit.
fn prefill_pass(
    plans: &[LayerPlan<'_>],
    rope: &RopeTable,
    kv_cache: &mut KvCache,
    geom: &PrefillGeometry,
    scratch: &mut PrefillScratch,
    range: PassRange,
    route: GemmRoute,
) -> ModelResult<()> {
    let m = range.rows;
    let h = geom.hidden;
    let inter = geom.intermediate;

    for (layer_idx, plan) in plans.iter().enumerate() {
        // ── attention norm (row-wise over the micro-batch) ───────────────
        row_norm(
            &scratch.hidden,
            &mut scratch.normed,
            m,
            h,
            plan.attn_norm_w,
            plan.eps,
        )?;

        // ── Q / K / V projections: one GEMM each over the micro-batch ────
        //
        // The three read the same input and write disjoint buffers, so they
        // run concurrently: one fork-join instead of three, and their tasks
        // share the pool (each GEMM's own arithmetic is unchanged).
        {
            let normed = &scratch.normed[..m * h];
            let (q_all, k_all, v_all) =
                (&mut scratch.q_all, &mut scratch.k_all, &mut scratch.v_all);
            let (q, (k, v)) = rayon::join(
                || plan.attn_q.gemm(route, normed, q_all, m, geom.q_dim, h),
                || {
                    rayon::join(
                        || plan.attn_k.gemm(route, normed, k_all, m, geom.kv_dim, h),
                        || plan.attn_v.gemm(route, normed, v_all, m, geom.kv_dim, h),
                    )
                },
            );
            q?;
            k?;
            v?;
        }

        // ── per position, in position order: QK-norm, RoPE, KV write ─────
        //
        // The writes stay strictly sequential: position `pos_start + i` is
        // stored before position `pos_start + i + 1`, exactly as the
        // per-token path stores them. Attention runs afterwards (below).
        for row in 0..m {
            let pos = range.pos_start + row;
            let hd = geom.head_dim;

            let q_row = &scratch.q_all[row * geom.q_dim..(row + 1) * geom.q_dim];
            let q_rope_row = &mut scratch.q_rope[row * geom.q_dim..(row + 1) * geom.q_dim];
            for head in 0..geom.num_heads {
                let span = head * hd..(head + 1) * hd;
                oxibonsai_kernels::rms_norm_simd(
                    &q_row[span.clone()],
                    plan.q_norm_w,
                    &mut scratch.q_normed[span.clone()],
                    plan.eps,
                )
                .map_err(ModelError::Kernel)?;
                rope.apply(&scratch.q_normed[span.clone()], &mut q_rope_row[span], pos)?;
            }

            let k_row = &scratch.k_all[row * geom.kv_dim..(row + 1) * geom.kv_dim];
            for head in 0..geom.num_kv_heads {
                let span = head * hd..(head + 1) * hd;
                oxibonsai_kernels::rms_norm_simd(
                    &k_row[span.clone()],
                    plan.k_norm_w,
                    &mut scratch.k_normed[span.clone()],
                    plan.eps,
                )
                .map_err(ModelError::Kernel)?;
                rope.apply(
                    &scratch.k_normed[span.clone()],
                    &mut scratch.k_rope[span],
                    pos,
                )?;
            }

            let v_row = &scratch.v_all[row * geom.kv_dim..(row + 1) * geom.kv_dim];
            for head in 0..geom.num_kv_heads {
                let span = head * hd..(head + 1) * hd;
                kv_cache.try_store_key(layer_idx, head, pos, &scratch.k_rope[span.clone()])?;
                kv_cache.try_store_value(layer_idx, head, pos, &v_row[span])?;
            }
            // Mirrors `block::functions::advance_kv_cache_to`: keep the
            // cache's own cursor at least one past the position just written.
            if kv_cache.seq_len() <= pos {
                kv_cache.set_seq_len(pos + 1);
            }
        }

        // ── causal GQA attention, every row of the micro-batch in parallel ─
        attend_rows(
            &scratch.q_rope[..m * geom.q_dim],
            &mut scratch.attn_out[..m * geom.q_dim],
            kv_cache,
            layer_idx,
            geom,
            range.pos_start,
        )?;

        // ── attention output projection + residual ───────────────────────
        plan.attn_output.gemm(
            route,
            &scratch.attn_out[..m * geom.q_dim],
            &mut scratch.proj,
            m,
            h,
            geom.q_dim,
        )?;
        add_into(&mut scratch.hidden, &scratch.proj, m * h);

        // ── FFN norm, gate/up GEMM, SwiGLU, down GEMM, residual ──────────
        row_norm(
            &scratch.hidden,
            &mut scratch.normed,
            m,
            h,
            plan.ffn_norm_w,
            plan.eps,
        )?;
        // Gate and up share their input too: one fork-join for both.
        {
            let normed = &scratch.normed[..m * h];
            let (gate, up) = (&mut scratch.gate, &mut scratch.up);
            let (g, u) = rayon::join(
                || plan.ffn_gate.gemm(route, normed, gate, m, inter, h),
                || plan.ffn_up.gemm(route, normed, up, m, inter, h),
            );
            g?;
            u?;
        }
        for row in 0..m {
            let span = row * inter..(row + 1) * inter;
            try_swiglu(
                &scratch.gate[span.clone()],
                &scratch.up[span.clone()],
                &mut scratch.swiglu[span],
            )?;
        }
        plan.ffn_down.gemm(
            route,
            &scratch.swiglu[..m * inter],
            &mut scratch.proj,
            m,
            h,
            inter,
        )?;
        add_into(&mut scratch.hidden, &scratch.proj, m * h);
    }
    Ok(())
}

/// Apply one RMSNorm weight to every row of a `[m x width]` matrix.
fn row_norm(
    input: &[f32],
    output: &mut [f32],
    m: usize,
    width: usize,
    weight: &[f32],
    eps: f32,
) -> ModelResult<()> {
    for (src, dst) in input[..m * width]
        .chunks_exact(width)
        .zip(output[..m * width].chunks_exact_mut(width))
    {
        oxibonsai_kernels::rms_norm_simd(src, weight, dst, eps).map_err(ModelError::Kernel)?;
    }
    Ok(())
}

/// `dst[..len] += src[..len]` (the residual add).
fn add_into(dst: &mut [f32], src: &[f32], len: usize) {
    for (d, s) in dst[..len].iter_mut().zip(src[..len].iter()) {
        *d += *s;
    }
}

/// Causal grouped-query attention of every row of one micro-batch, in
/// parallel over rows.
///
/// Row `r` (absolute position `pos_start + r`) attends over exactly
/// positions `0..=pos_start + r`: it calls the very function the per-block
/// decode path runs ([`crate::block::functions::gqa_attention`]) with
/// `seq_len = pos_start + r + 1`, and [`KvCache::attend_group`] reads exactly
/// the first `seq_len` stored positions — never the cursor, never past
/// `seq_len`. So running the rows concurrently, after the whole micro-batch
/// has been stored, gives every row the bits it would get one at a time: the
/// later rows' keys and values are in the cache but outside every earlier
/// row's window. The batched prefill and the sequential decode therefore
/// still read the cache — `f32` or `f16` — through one body and cannot
/// drift. The cache's allocation cannot change between the stores and these
/// reads: every position of the call was made resident up front
/// (`ensure_context_capacity(last_pos)`).
///
/// With fewer rows than worker threads, each row's KV heads also run in
/// parallel (the per-token path's own setting); with more, whole rows are
/// the unit of work. Either way the per-head arithmetic is identical.
fn attend_rows(
    q_rope: &[f32],
    attn_out: &mut [f32],
    kv_cache: &KvCache,
    layer_idx: usize,
    geom: &PrefillGeometry,
    pos_start: usize,
) -> ModelResult<()> {
    let q_dim = geom.q_dim;
    // `prefill_geometry` never yields a zero head width; checked anyway so a
    // zero can never reach `par_chunks_mut`.
    let rows = q_rope
        .len()
        .checked_div(q_dim)
        .ok_or_else(|| ModelError::ShapeInvariant {
            tensor: format!("layer {layer_idx}: batched attention"),
            expected: "q_dim >= 1".to_string(),
            actual: "q_dim = 0".to_string(),
        })?;
    let heads_parallel = rows < rayon::current_num_threads();
    attn_out
        .par_chunks_mut(q_dim)
        .zip(q_rope.par_chunks(q_dim))
        .enumerate()
        .try_for_each(|(row, (out_row, q_row))| {
            crate::block::functions::gqa_attention(
                q_row,
                out_row,
                kv_cache,
                layer_idx,
                geom.num_heads,
                geom.heads_per_group,
                geom.head_dim,
                pos_start + row + 1,
                heads_parallel,
            )
        })
}

/// The batched prefill's tests, and the fixtures `forward_hidden`'s tests
/// share with them (`pub(in crate::model::types)`: test-only, and only for
/// this module's siblings).
#[cfg(test)]
#[path = "prefill_cpu_tests.rs"]
pub(super) mod tests;
