//! Individual Metal kernel dispatch methods for `MetalGraph`.
//!
//! Each `dispatch_*` method encodes a single GPU kernel invocation
//! into the currently active compute command encoder.

#![cfg(feature = "metal")]

use metal::{Buffer, ComputePipelineState, MTLSize};

use super::metal_graph::{div_ceil, set_scalar, MetalGraph, MetalGraphError};

/// Largest `head_dim` the `batched_attention_scores_v2` MSL kernel computes a
/// *complete* dot product for.
///
/// The kernel stages the query row into `threadgroup float shared_q[128]` and
/// accumulates over exactly 128 dims with no bound check
/// (`kernel_sources/attention.rs`), so a larger `head_dim` silently scores on
/// the first 128 dims — measured on this M3 as 4.13628 vs the CPU's 7.43115 at
/// `head_dim 256`, with `MTLCommandBufferStatus == Completed` (MET-01).
///
/// Every shipping model here uses `head_dim` 64 or 128. Bonsai 2 uses 256, so
/// B2-15 makes the kernel `head_dim`-generic (staging sized to
/// `MAX_HEAD_DIM = 256`) and raises this constant with it.
pub(crate) const ATTENTION_SCORES_V2_MAX_HEAD_DIM: u32 = 128;

/// Largest `k` [`MetalGraph::dispatch_topk_f32`] accepts.
///
/// The `topk_f32` MSL kernel (`kernel_sources/utility.rs`) tracks its `k`
/// already-picked winners in a fixed-size `threadgroup uint picked[256]`
/// array, so this bound is a hard ceiling shared by both sides: the kernel
/// itself additionally clamps to it in case a caller ever bypasses this
/// dispatcher, but `dispatch_topk_f32` rejects a larger `k` outright with a
/// typed error rather than silently truncating the caller's request to
/// fewer winners than asked for. 256 comfortably covers every realistic
/// `top_k` sampling value (Bonsai 2's recommended sampling config uses
/// `top_k = 20`).
///
/// Wiring a sampled decode request through `dispatch_topk_f32` is
/// METAL-CONCURRENCY's (wave 4; see the `topk_f32` field doc on
/// `MetalPipelines`), so nothing in the non-test build reads this constant
/// yet — this crate's own tests do, hence `#[allow(dead_code)]`.
#[allow(dead_code)]
pub(crate) const MAX_TOPK_F32: u32 = 256;

impl MetalGraph {
    // ─────────────────────────────────────────────────────────────────────
    // Internal: individual kernel dispatch (all use the shared encoder)
    // ─────────────────────────────────────────────────────────────────────

    /// Dispatch `gemv_q1_g128_v7` (single-row, fully unrolled) into the given encoder.
    ///
    /// V7: 8 simdgroups × 1 row = 8 rows per threadgroup.
    /// Fully unrolled inner loop for maximum instruction-level parallelism.
    ///
    /// Buffer layout:
    /// - buffer(0) = blocks_raw (u8 weight data, SoA layout)
    /// - buffer(1) = input (f32, read as float4* by the kernel)
    /// - buffer(2) = output (f32)
    /// - buffer(3) = n_rows (u32, set_bytes)
    /// - buffer(4) = k (u32, set_bytes)
    ///
    /// Dispatch: `[ceil(n_rows/8), 1, 1]` threadgroups, `[256, 1, 1]` threads
    pub(crate) fn dispatch_gemv_q1(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        blocks: &Buffer,
        input: &Buffer,
        output: &Buffer,
        n_rows: u32,
        k: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.pipelines.gemv_q1_g128_v7);
        encoder.set_buffer(0, Some(blocks), 0);
        encoder.set_buffer(1, Some(input), 0);
        encoder.set_buffer(2, Some(output), 0);
        unsafe {
            set_scalar(encoder, 3, &n_rows);
            set_scalar(encoder, 4, &k);
        }

        let tg_count = div_ceil(n_rows as usize, 8);
        encoder
            .dispatch_thread_groups(MTLSize::new(tg_count as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    /// Dispatch `gemv_tq2_g128_v1` (SIMD-group-per-row) into the given encoder.
    ///
    /// Identical threading shape to `gemv_q1_g128_v7` (8 rows/threadgroup,
    /// 256 threads), but operates on TQ2_0_g128 (ternary) weights in SoA
    /// layout `[N×2B scales][N×32B qs]`.
    ///
    /// Buffer layout:
    /// - buffer(0) = soa_raw (u8 SoA TQ2 weights)
    /// - buffer(1) = input (f32, read as float4* by the kernel)
    /// - buffer(2) = output (f32)
    /// - buffer(3) = n_rows (u32, set_bytes)
    /// - buffer(4) = k (u32, set_bytes)
    ///
    /// Dispatch: `[ceil(n_rows/8), 1, 1]` threadgroups, `[256, 1, 1]` threads.
    pub(crate) fn dispatch_gemv_tq2(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        blocks: &Buffer,
        input: &Buffer,
        output: &Buffer,
        n_rows: u32,
        k: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.pipelines.gemv_tq2_g128_v1);
        encoder.set_buffer(0, Some(blocks), 0);
        encoder.set_buffer(1, Some(input), 0);
        encoder.set_buffer(2, Some(output), 0);
        unsafe {
            set_scalar(encoder, 3, &n_rows);
            set_scalar(encoder, 4, &k);
        }

        let tg_count = div_ceil(n_rows as usize, 8);
        encoder
            .dispatch_thread_groups(MTLSize::new(tg_count as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    /// Dispatch fused GEMV + residual add: `output[row] = residual[row] + gemv(blocks, input)[row]`.
    ///
    /// V7 single-row: fully unrolled inner loop.
    /// Eliminates a separate `residual_add` dispatch by folding the add into
    /// the GEMV's final write.  `output` and `residual` may alias.
    ///
    /// Buffer layout:
    /// - buffer(0) = blocks_raw (u8 weight data, SoA layout)
    /// - buffer(1) = input (f32, read as float4*)
    /// - buffer(2) = output (f32, written: residual + gemv_result)
    /// - buffer(3) = n_rows (u32, set_bytes)
    /// - buffer(4) = k (u32, set_bytes)
    /// - buffer(5) = residual (f32)
    ///
    /// Dispatch: `[ceil(n_rows/8), 1, 1]` threadgroups, `[256, 1, 1]` threads
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_gemv_q1_residual(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        blocks: &Buffer,
        input: &Buffer,
        output: &Buffer,
        n_rows: u32,
        k: u32,
        residual: &Buffer,
    ) {
        encoder.set_compute_pipeline_state(&self.pipelines.gemv_q1_g128_v7_residual);
        encoder.set_buffer(0, Some(blocks), 0);
        encoder.set_buffer(1, Some(input), 0);
        encoder.set_buffer(2, Some(output), 0);
        unsafe {
            set_scalar(encoder, 3, &n_rows);
            set_scalar(encoder, 4, &k);
        }
        encoder.set_buffer(5, Some(residual), 0);

        let tg_count = div_ceil(n_rows as usize, 8);
        encoder
            .dispatch_thread_groups(MTLSize::new(tg_count as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    /// Dispatch `rmsnorm_weighted_v2` (parallel reduction) into the given encoder.
    ///
    /// V2 uses a single threadgroup of 256 threads with cooperative
    /// shared-memory reduction to compute sum-of-squares in O(n) total
    /// work, fixing the O(n²) issue in V1.
    ///
    /// Buffer layout:
    /// - buffer(0) = input (f32)
    /// - buffer(1) = weight (f32)
    /// - buffer(2) = output (f32)
    /// - buffer(3) = eps (f32, set_bytes)
    /// - buffer(4) = n (u32, set_bytes)
    ///
    /// Dispatch: `[1, 1, 1]` threadgroups, `[256, 1, 1]` threads
    pub(crate) fn dispatch_rmsnorm(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        input: &Buffer,
        weight: &Buffer,
        output: &Buffer,
        eps: f32,
        n: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.pipelines.rmsnorm_weighted_v2);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(weight), 0);
        encoder.set_buffer(2, Some(output), 0);
        unsafe {
            set_scalar(encoder, 3, &eps);
            set_scalar(encoder, 4, &n);
        }

        // Single threadgroup processes the entire vector cooperatively
        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
    }

    /// Dispatch fused gate+up+SwiGLU kernel.
    ///
    /// Combines the separate gate_up GEMV and SwiGLU dispatches into one.
    /// Each simdgroup computes both gate[pos] and up[pos] from the
    /// row-concatenated weight buffer, then applies `silu(gate) * up`.
    ///
    /// Buffer layout:
    /// - buffer(0) = blocks_raw (u8, gate+up weights — rows [0..inter) = gate, [inter..2*inter) = up)
    /// - buffer(1) = input (f32, normed hidden state, read as float4*)
    /// - buffer(2) = output (f32, swiglu result `[inter_size]`)
    /// - buffer(3) = inter_size (u32, set_bytes)
    /// - buffer(4) = k (u32, set_bytes — hidden_size)
    ///
    /// Dispatch: `[ceil(inter_size/8), 1, 1]` threadgroups, `[256, 1, 1]` threads
    pub(crate) fn dispatch_fused_gate_up_swiglu(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        weight_buf: &Buffer,
        input_buf: &Buffer,
        output_buf: &Buffer,
        inter_size: u32,
        k: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.pipelines.fused_gate_up_swiglu_q1);
        encoder.set_buffer(0, Some(weight_buf), 0);
        encoder.set_buffer(1, Some(input_buf), 0);
        encoder.set_buffer(2, Some(output_buf), 0);
        unsafe {
            set_scalar(encoder, 3, &inter_size);
            set_scalar(encoder, 4, &k);
        }

        let tg_count = div_ceil(inter_size as usize, 8);
        encoder
            .dispatch_thread_groups(MTLSize::new(tg_count as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    // ─────────────────────────────────────────────────────────────────────
    // V7-based GEMM dispatch methods (batch prefill)
    // ─────────────────────────────────────────────────────────────────────

    /// Dispatch V7-based GEMM: `outputs[col][row] = dot(weights[row], inputs[col])`.
    ///
    /// Column-major layout: `inputs[col * k + elem]`, `outputs[col * n_rows + row]`.
    /// 1D grid: `[ceil(n_rows/8), 1, 1]` threadgroups — batch columns processed inside kernel.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_gemm_q1_v7(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        blocks: &Buffer,
        inputs: &Buffer,
        outputs: &Buffer,
        n_rows: u32,
        k: u32,
        batch_size: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.pipelines.gemm_q1_g128_v7);
        encoder.set_buffer(0, Some(blocks), 0);
        encoder.set_buffer(1, Some(inputs), 0);
        encoder.set_buffer(2, Some(outputs), 0);
        unsafe {
            set_scalar(encoder, 3, &n_rows);
            set_scalar(encoder, 4, &batch_size);
            set_scalar(encoder, 5, &k);
        }

        let tg_x = div_ceil(n_rows as usize, 8) as u64;
        encoder.dispatch_thread_groups(MTLSize::new(tg_x, 1, 1), MTLSize::new(256, 1, 1));
    }

    /// Dispatch V7-based GEMM with residual addition:
    /// `outputs[col][row] = residual[col][row] + dot(weights[row], inputs[col])`.
    ///
    /// `outputs` and `residual` may alias (in-place residual add).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_gemm_q1_v7_residual(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        blocks: &Buffer,
        inputs: &Buffer,
        outputs: &Buffer,
        n_rows: u32,
        k: u32,
        batch_size: u32,
        residual: &Buffer,
    ) {
        encoder.set_compute_pipeline_state(&self.pipelines.gemm_q1_g128_v7_residual);
        encoder.set_buffer(0, Some(blocks), 0);
        encoder.set_buffer(1, Some(inputs), 0);
        encoder.set_buffer(2, Some(outputs), 0);
        unsafe {
            set_scalar(encoder, 3, &n_rows);
            set_scalar(encoder, 4, &batch_size);
            set_scalar(encoder, 5, &k);
        }
        encoder.set_buffer(6, Some(residual), 0);

        let tg_x = div_ceil(n_rows as usize, 8) as u64;
        encoder.dispatch_thread_groups(MTLSize::new(tg_x, 1, 1), MTLSize::new(256, 1, 1));
    }

    /// Dispatch V7-style **TQ2** GEMM (batched ternary): `outputs[col][row] = decode_tq2(weights[row]) · inputs[col]`.
    ///
    /// Column-major layout: `inputs[col * k + elem]`, `outputs[col * n_rows + row]`.
    /// 1D grid: `[ceil(n_rows/8), 1, 1]` threadgroups — batch columns processed
    /// inside the kernel via 8-column outer chunks (handles arbitrary
    /// `batch_size` correctly).
    ///
    /// Buffer layout matches `dispatch_gemv_tq2`'s SoA conventions:
    /// `[N×2B FP16 scales][N×32B 2-bit qs]`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_gemm_tq2_v7(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        blocks: &Buffer,
        inputs: &Buffer,
        outputs: &Buffer,
        n_rows: u32,
        k: u32,
        batch_size: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.pipelines.gemm_tq2_g128_v7);
        encoder.set_buffer(0, Some(blocks), 0);
        encoder.set_buffer(1, Some(inputs), 0);
        encoder.set_buffer(2, Some(outputs), 0);
        unsafe {
            set_scalar(encoder, 3, &n_rows);
            set_scalar(encoder, 4, &batch_size);
            set_scalar(encoder, 5, &k);
        }

        let tg_x = div_ceil(n_rows as usize, 8) as u64;
        encoder.dispatch_thread_groups(MTLSize::new(tg_x, 1, 1), MTLSize::new(256, 1, 1));
    }

    /// Dispatch tiled **TQ2** GEMM (`v8`) for the **large-M** path (DiT).
    ///
    /// Same op and column-major buffer layout as [`dispatch_gemm_tq2_v7`]
    /// (`inputs[col*k + elem]`, `outputs[col*n_rows + row]`) and reads the
    /// *same* SoA weight buffer, but uses a **2-D grid**
    /// `[ceil(N/8), ceil(M/32), 1]` with `256`-thread (`8×32`) register-blocked
    /// micro-tiles (`TN=8`, `TM=32`, `TK=128`). This parallelizes the batch `M`
    /// across threadgroups and decodes each weight block once per M-tile,
    /// instead of `v7`'s single-row grid that walks `M` serially in 8-column
    /// chunks (re-decoding weights `M/8` times). Numerically equivalent to `v7`.
    ///
    /// The `TN` / `TM` / `TK` tile constants here MUST match the
    /// `gemm_tq2_g128_v8_tiled` MSL kernel.
    ///
    /// Retained as a fallback now that `encode_gemm_tq2` dispatches the faster
    /// `simdgroup_matrix` [`Self::dispatch_gemm_tq2_v9`]; still exercised by the
    /// v8/v9 parity + ratio benchmarks (hence `#[allow(dead_code)]` for
    /// non-test builds).
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_gemm_tq2_v8(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        blocks: &Buffer,
        inputs: &Buffer,
        outputs: &Buffer,
        n_rows: u32,
        k: u32,
        batch_size: u32,
    ) {
        // Tile sizes — keep in sync with MSL_GEMM_TQ2_G128_V8_TILED.
        // Register-blocked: V8_RN=4 output rows per thread, so
        // threads = (TN/RN) * TM = (32/4) * 16 = 128.
        const TN: usize = 32; // weight rows per threadgroup (grid.x)
        const TM: usize = 16; // batch columns per threadgroup (grid.y)
        const RN: usize = 4; // output rows accumulated per thread
        const THREADS: u64 = ((TN / RN) * TM) as u64; // 128

        encoder.set_compute_pipeline_state(&self.pipelines.gemm_tq2_g128_v8_tiled);
        encoder.set_buffer(0, Some(blocks), 0);
        encoder.set_buffer(1, Some(inputs), 0);
        encoder.set_buffer(2, Some(outputs), 0);
        unsafe {
            set_scalar(encoder, 3, &n_rows);
            set_scalar(encoder, 4, &batch_size);
            set_scalar(encoder, 5, &k);
        }

        let tg_x = div_ceil(n_rows as usize, TN) as u64;
        let tg_y = div_ceil(batch_size as usize, TM) as u64;
        encoder.dispatch_thread_groups(MTLSize::new(tg_x, tg_y, 1), MTLSize::new(THREADS, 1, 1));
    }

    /// Dispatch `simdgroup_matrix` **TQ2** GEMM (`v9`) for the **large-M** path.
    ///
    /// Same op and column-major buffer layout as [`dispatch_gemm_tq2_v7`] /
    /// [`Self::dispatch_gemm_tq2_v8`] (`inputs[col*k + elem]`,
    /// `outputs[col*n_rows + row]`) and reads the *same* SoA weight buffer, but
    /// computes `C = A · Dᵀ` with Apple's `simdgroup_float8x8` 8×8×8 hardware
    /// MAC units. Each threadgroup owns a `V9_TM × V9_TN = 64 × 64` output tile
    /// (4 simdgroups, 128 threads), K-tiled by `V9_TK = 32`, dequantizing the
    /// weight transposed into threadgroup memory once per K-tile and reusing it
    /// across all 64 `M` columns via the matrix units. Numerically equivalent
    /// to `v7` / `v8` (f32 accumulate; f16-staged operands).
    ///
    /// The `V9_TM` / `V9_TN` / `V9_TK` tile constants and the `4`-simdgroup
    /// (`128`-thread) shape here MUST match the `gemm_tq2_g128_v9_simdgroup`
    /// MSL kernel.
    ///
    /// Retained as a fallback now that `encode_gemm_tq2` dispatches the
    /// staging-optimized [`Self::dispatch_gemm_tq2_v10`] (~3.86× faster on the
    /// big DiT shapes); still exercised by the v9/v10 parity + ratio benchmarks
    /// (hence `#[allow(dead_code)]` for non-test builds).
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_gemm_tq2_v9(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        blocks: &Buffer,
        inputs: &Buffer,
        outputs: &Buffer,
        n_rows: u32,
        k: u32,
        batch_size: u32,
    ) {
        // Tile sizes / simdgroup shape — keep in sync with
        // MSL_GEMM_TQ2_G128_V9_SIMDGROUP.
        const TN: usize = 64; // weight rows per threadgroup (grid.x)
        const TM: usize = 64; // batch columns per threadgroup (grid.y)
        const SIMDGROUPS: u64 = 4; // 32x32 quadrant each -> 64x64 tile
        const THREADS: u64 = SIMDGROUPS * 32; // 128

        encoder.set_compute_pipeline_state(&self.pipelines.gemm_tq2_g128_v9_simdgroup);
        encoder.set_buffer(0, Some(blocks), 0);
        encoder.set_buffer(1, Some(inputs), 0);
        encoder.set_buffer(2, Some(outputs), 0);
        unsafe {
            set_scalar(encoder, 3, &n_rows);
            set_scalar(encoder, 4, &batch_size);
            set_scalar(encoder, 5, &k);
        }

        let tg_x = div_ceil(n_rows as usize, TN) as u64;
        let tg_y = div_ceil(batch_size as usize, TM) as u64;
        encoder.dispatch_thread_groups(MTLSize::new(tg_x, tg_y, 1), MTLSize::new(THREADS, 1, 1));
    }

    /// Dispatch staging-optimized `simdgroup_matrix` **TQ2** GEMM (`v10`) for the
    /// **large-M** path (DiT).
    ///
    /// Same op, column-major buffer layout, SoA weight buffer, and `64×64`
    /// output-tile / `4`-simdgroup (`128`-thread) shape as
    /// [`Self::dispatch_gemm_tq2_v9`], so the grid is identical
    /// (`[ceil(N/64), ceil(M/64), 1]`). The kernel differs only in *how it
    /// stages* each K-tile: it dequantizes the weight into threadgroup memory as
    /// `half` (exact for ternary `code×scale`), spreads the dequant-scatter
    /// across all 128 threads with vectorized `uint` qs loads, and
    /// double-buffers the K-tile staging to overlap the staging latency with the
    /// 8×8 matrix MACs. Numerically equivalent to `v7`/`v8`/`v9` (f32 accumulate;
    /// `A` staged f32, `D` staged half).
    ///
    /// The `V10_TM` / `V10_TN` / `V10_TK` tile constants and the `4`-simdgroup
    /// (`128`-thread) shape here MUST match the `gemm_tq2_g128_v10_simdgroup`
    /// MSL kernel.
    ///
    /// This is the kernel `encode_gemm_tq2` now dispatches for the DiT large-M
    /// path (it passed the v9/v10 parity sweep and beat `v9` ~3.86× on the big
    /// DiT shapes).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_gemm_tq2_v10(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        blocks: &Buffer,
        inputs: &Buffer,
        outputs: &Buffer,
        n_rows: u32,
        k: u32,
        batch_size: u32,
    ) {
        // Tile sizes / simdgroup shape — keep in sync with
        // MSL_GEMM_TQ2_G128_V10_SIMDGROUP.
        const TN: usize = 64; // weight rows per threadgroup (grid.x)
        const TM: usize = 64; // batch columns per threadgroup (grid.y)
        const SIMDGROUPS: u64 = 4; // 32x32 quadrant each -> 64x64 tile
        const THREADS: u64 = SIMDGROUPS * 32; // 128

        encoder.set_compute_pipeline_state(&self.pipelines.gemm_tq2_g128_v10_simdgroup);
        encoder.set_buffer(0, Some(blocks), 0);
        encoder.set_buffer(1, Some(inputs), 0);
        encoder.set_buffer(2, Some(outputs), 0);
        unsafe {
            set_scalar(encoder, 3, &n_rows);
            set_scalar(encoder, 4, &batch_size);
            set_scalar(encoder, 5, &k);
        }

        let tg_x = div_ceil(n_rows as usize, TN) as u64;
        let tg_y = div_ceil(batch_size as usize, TM) as u64;
        encoder.dispatch_thread_groups(MTLSize::new(tg_x, tg_y, 1), MTLSize::new(THREADS, 1, 1));
    }

    /// Dispatch the f32-exact `simdgroup_matrix` GEMM (`gemm_f32_simdgroup`) for
    /// the large-M **text-encoder** path (Qwen3-4B).
    ///
    /// Computes `out[M,N] = A[M,K] · W[N,K]ᵀ` over **pure-f32** weights, with the
    /// same column-major buffer layout as the ternary
    /// [`Self::dispatch_gemm_tq2_v9`] (`inputs[col*k + elem]`,
    /// `outputs[col*n_rows + row]`) and the same `64×64` output-tile /
    /// `4`-simdgroup (`128`-thread) shape, so the grid is identical
    /// (`[ceil(N/64), ceil(M/64), 1]`). The only difference from `v9` is
    /// buffer(0): a plain row-major f32 weight buffer (`weights[n*k + elem]`)
    /// instead of the SoA ternary block buffer — there is no scale section and
    /// no dequant. Numerically equivalent to the CPU `gemm_abt` (cos ≈ 1.0).
    ///
    /// The `F32_TM` / `F32_TN` / `F32_TK` tile constants and the `4`-simdgroup
    /// (`128`-thread) shape here MUST match the `gemm_f32_simdgroup` MSL kernel.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_gemm_f32(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        weights: &Buffer,
        inputs: &Buffer,
        outputs: &Buffer,
        n_rows: u32,
        k: u32,
        batch_size: u32,
    ) {
        // Tile sizes / simdgroup shape — keep in sync with MSL_GEMM_F32_SIMDGROUP.
        const TN: usize = 64; // weight rows per threadgroup (grid.x)
        const TM: usize = 64; // batch columns per threadgroup (grid.y)
        const SIMDGROUPS: u64 = 4; // 32x32 quadrant each -> 64x64 tile
        const THREADS: u64 = SIMDGROUPS * 32; // 128

        encoder.set_compute_pipeline_state(&self.pipelines.gemm_f32_simdgroup);
        encoder.set_buffer(0, Some(weights), 0);
        encoder.set_buffer(1, Some(inputs), 0);
        encoder.set_buffer(2, Some(outputs), 0);
        unsafe {
            set_scalar(encoder, 3, &n_rows);
            set_scalar(encoder, 4, &batch_size);
            set_scalar(encoder, 5, &k);
        }

        let tg_x = div_ceil(n_rows as usize, TN) as u64;
        let tg_y = div_ceil(batch_size as usize, TM) as u64;
        encoder.dispatch_thread_groups(MTLSize::new(tg_x, tg_y, 1), MTLSize::new(THREADS, 1, 1));
    }

    /// Dispatch the bf16-input / f32-accumulate `gemm_bf16_simdgroup` GEMM. Same
    /// buffers, scalars, tile shape, and grid as [`Self::dispatch_gemm_f32`] —
    /// only the pipeline (and thus the internal staging precision) differs.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_gemm_bf16(
        &self,
        pso: &metal::ComputePipelineState,
        encoder: &metal::ComputeCommandEncoderRef,
        weights: &Buffer,
        inputs: &Buffer,
        outputs: &Buffer,
        n_rows: u32,
        k: u32,
        batch_size: u32,
    ) {
        // Tile sizes / simdgroup shape — keep in sync with MSL_GEMM_BF16_SIMDGROUP.
        const TN: usize = 64;
        const TM: usize = 64;
        const SIMDGROUPS: u64 = 4;
        const THREADS: u64 = SIMDGROUPS * 32; // 128

        encoder.set_compute_pipeline_state(pso);
        encoder.set_buffer(0, Some(weights), 0);
        encoder.set_buffer(1, Some(inputs), 0);
        encoder.set_buffer(2, Some(outputs), 0);
        unsafe {
            set_scalar(encoder, 3, &n_rows);
            set_scalar(encoder, 4, &batch_size);
            set_scalar(encoder, 5, &k);
        }

        let tg_x = div_ceil(n_rows as usize, TN) as u64;
        let tg_y = div_ceil(batch_size as usize, TM) as u64;
        encoder.dispatch_thread_groups(MTLSize::new(tg_x, tg_y, 1), MTLSize::new(THREADS, 1, 1));
    }

    /// Dispatch fused gate+up+SwiGLU GEMM for batch prefill.
    ///
    /// 1D grid: `[ceil(inter_size/8), 1, 1]` threadgroups — batch columns processed inside kernel.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_fused_gate_up_swiglu_gemm(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        weight_buf: &Buffer,
        inputs: &Buffer,
        outputs: &Buffer,
        inter_size: u32,
        k: u32,
        batch_size: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.pipelines.fused_gate_up_swiglu_gemm_q1);
        encoder.set_buffer(0, Some(weight_buf), 0);
        encoder.set_buffer(1, Some(inputs), 0);
        encoder.set_buffer(2, Some(outputs), 0);
        unsafe {
            set_scalar(encoder, 3, &inter_size);
            set_scalar(encoder, 4, &batch_size);
            set_scalar(encoder, 5, &k);
        }

        let tg_x = div_ceil(inter_size as usize, 8) as u64;
        encoder.dispatch_thread_groups(MTLSize::new(tg_x, 1, 1), MTLSize::new(256, 1, 1));
    }

    /// Dispatch `residual_add` into the given encoder (in-place on `a`).
    ///
    /// Buffer layout:
    /// - buffer(0) = a (f32, read-write, modified in-place)
    /// - buffer(1) = b (f32)
    /// - buffer(2) = n (u32, set_bytes)
    ///
    /// Dispatch: [ceil(n/256), 1, 1] threadgroups, [256, 1, 1] threads
    pub(crate) fn dispatch_residual_add(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        a: &Buffer,
        b: &Buffer,
        n: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.pipelines.residual_add);
        encoder.set_buffer(0, Some(a), 0);
        encoder.set_buffer(1, Some(b), 0);
        unsafe {
            set_scalar(encoder, 2, &n);
        }

        let tg_count = div_ceil(n as usize, 256);
        encoder
            .dispatch_thread_groups(MTLSize::new(tg_count as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    // ─────────────────────────────────────────────────────────────────────
    // FLUX.2 VAE decoder per-op f32 primitives
    // ─────────────────────────────────────────────────────────────────────

    /// Dispatch `im2col_f32` for a tile of output rows `[row_start, row_start +
    /// tile_rows)`, writing `patches[tile_rows, patch_dim]` in `(kH,kW,C_in)`
    /// order (`patch_dim = k·k·c_in`). One thread per patch element.
    ///
    /// Buffer layout matches `MSL_IM2COL_F32` (input, patches, then the scalars
    /// `c_in,h,w,k,pad,w_out,row_start,n_elems`). `n_elems = tile_rows *
    /// patch_dim` is the grid bound.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_im2col_f32(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        input: &Buffer,
        patches: &Buffer,
        c_in: u32,
        h: u32,
        w: u32,
        k: u32,
        pad: u32,
        w_out: u32,
        row_start: u32,
        n_elems: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.pipelines.im2col_f32);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(patches), 0);
        unsafe {
            set_scalar(encoder, 2, &c_in);
            set_scalar(encoder, 3, &h);
            set_scalar(encoder, 4, &w);
            set_scalar(encoder, 5, &k);
            set_scalar(encoder, 6, &pad);
            set_scalar(encoder, 7, &w_out);
            set_scalar(encoder, 8, &row_start);
            set_scalar(encoder, 9, &n_elems);
        }
        let tg_count = div_ceil(n_elems as usize, 256);
        encoder
            .dispatch_thread_groups(MTLSize::new(tg_count as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    /// Dispatch the **im2col-free implicit-GEMM** Conv2d `conv2d_f32_implicit`:
    /// `out[C_out, P] = weight[C_out, kk_cin] · Patches[kk_cin, P]` with the conv
    /// patches gathered on-the-fly into threadgroup memory (no global im2col).
    /// `P = H_out·W_out`, `kk_cin = k·k·C_in`. Output is row-major NCHW
    /// `[C_out, P]` (NO bias — the host adds it on download).
    ///
    /// Buffer layout matches `MSL_CONV2D_F32_IMPLICIT` (weight, input, output,
    /// then the scalars `c_out, p, kk_cin, c_in, h, w, k, pad, w_out`).
    ///
    /// Tile geometry mirrors `dispatch_gemm_f32` (64×64 tile, 4 simdgroups, 128
    /// threads): grid `[ceil(P/64), ceil(C_out/64), 1]`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_conv2d_f32_implicit(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        weight: &Buffer,
        input: &Buffer,
        output: &Buffer,
        c_out: u32,
        p: u32,
        kk_cin: u32,
        c_in: u32,
        h: u32,
        w: u32,
        k: u32,
        pad: u32,
        w_out: u32,
    ) {
        // Tile sizes / simdgroup shape — keep in sync with MSL_CONV2D_F32_IMPLICIT.
        const TN: usize = 64; // output pixels per threadgroup (grid.x, N = P)
        const TM: usize = 64; // output channels per threadgroup (grid.y, M = C_out)
        const SIMDGROUPS: u64 = 4; // 32x32 quadrant each -> 64x64 tile
        const THREADS: u64 = SIMDGROUPS * 32; // 128

        encoder.set_compute_pipeline_state(&self.pipelines.conv2d_f32_implicit);
        encoder.set_buffer(0, Some(weight), 0);
        encoder.set_buffer(1, Some(input), 0);
        encoder.set_buffer(2, Some(output), 0);
        unsafe {
            set_scalar(encoder, 3, &c_out);
            set_scalar(encoder, 4, &p);
            set_scalar(encoder, 5, &kk_cin);
            set_scalar(encoder, 6, &c_in);
            set_scalar(encoder, 7, &h);
            set_scalar(encoder, 8, &w);
            set_scalar(encoder, 9, &k);
            set_scalar(encoder, 10, &pad);
            set_scalar(encoder, 11, &w_out);
        }

        let tg_x = div_ceil(p as usize, TN) as u64;
        let tg_y = div_ceil(c_out as usize, TM) as u64;
        encoder.dispatch_thread_groups(MTLSize::new(tg_x, tg_y, 1), MTLSize::new(THREADS, 1, 1));
    }

    /// Dispatch `groupnorm_f32` (in-place on `x`, NCHW `[channels, hw]`): one
    /// threadgroup per group, 256 threads, Kahan-compensated f32 reduction.
    ///
    /// Buffer layout matches `MSL_GROUPNORM_F32` (x, weight, bias, then the
    /// scalars `channels, hw, num_groups, eps`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_groupnorm_f32(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        x: &Buffer,
        weight: &Buffer,
        bias: &Buffer,
        channels: u32,
        hw: u32,
        num_groups: u32,
        eps: f32,
    ) {
        encoder.set_compute_pipeline_state(&self.pipelines.groupnorm_f32);
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_buffer(1, Some(weight), 0);
        encoder.set_buffer(2, Some(bias), 0);
        unsafe {
            set_scalar(encoder, 3, &channels);
            set_scalar(encoder, 4, &hw);
            set_scalar(encoder, 5, &num_groups);
            set_scalar(encoder, 6, &eps);
        }
        // One threadgroup per group; 256 threads cooperatively reduce the group.
        encoder.dispatch_thread_groups(
            MTLSize::new(num_groups as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    }

    /// Dispatch `silu_f32` (in-place, element-wise `x / (1 + exp(-x))`).
    ///
    /// Buffer layout matches `MSL_SILU_F32` (x, then scalar `n`).
    pub(crate) fn dispatch_silu_f32(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        x: &Buffer,
        n: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.pipelines.silu_f32);
        encoder.set_buffer(0, Some(x), 0);
        unsafe {
            set_scalar(encoder, 1, &n);
        }
        let tg_count = div_ceil(n as usize, 256);
        encoder
            .dispatch_thread_groups(MTLSize::new(tg_count as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    /// Dispatch `upsample_nearest_f32` (`[c, h, w] → [c, 2h, 2w]`, one thread per
    /// output element).
    ///
    /// Buffer layout matches `MSL_UPSAMPLE_NEAREST_F32` (input, output, then the
    /// scalars `c, h, w, n_out`). `n_out = c * 2h * 2w` is the grid bound.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_upsample_nearest_f32(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        input: &Buffer,
        output: &Buffer,
        c: u32,
        h: u32,
        w: u32,
        n_out: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.pipelines.upsample_nearest_f32);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(output), 0);
        unsafe {
            set_scalar(encoder, 2, &c);
            set_scalar(encoder, 3, &h);
            set_scalar(encoder, 4, &w);
            set_scalar(encoder, 5, &n_out);
        }
        let tg_count = div_ceil(n_out as usize, 256);
        encoder
            .dispatch_thread_groups(MTLSize::new(tg_count as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    // ─────────────────────────────────────────────────────────────────────
    // Fused dispatch helpers (reduce 6 dispatches → 3 per attention sublayer)
    // ─────────────────────────────────────────────────────────────────────

    /// Dispatch `fused_qk_norm`: RMSNorm both Q and K heads in one dispatch.
    ///
    /// Replaces two separate `batched_rmsnorm_v2` dispatches.
    ///
    /// Dispatch: `[nq + nkv, 1, 1]` threadgroups, `[256, 1, 1]` threads
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_fused_qk_norm(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        q_in: &Buffer,
        q_in_offset: u64,
        k_in: &Buffer,
        k_in_offset: u64,
        q_out: &Buffer,
        k_out: &Buffer,
        q_weight: &Buffer,
        k_weight: &Buffer,
        nq: u32,
        nkv: u32,
        head_dim: u32,
        eps: f32,
    ) {
        encoder.set_compute_pipeline_state(&self.pipelines.fused_qk_norm);
        encoder.set_buffer(0, Some(q_in), q_in_offset);
        encoder.set_buffer(1, Some(k_in), k_in_offset);
        encoder.set_buffer(2, Some(q_out), 0);
        encoder.set_buffer(3, Some(k_out), 0);
        encoder.set_buffer(4, Some(q_weight), 0);
        encoder.set_buffer(5, Some(k_weight), 0);
        unsafe {
            set_scalar(encoder, 6, &nq);
            set_scalar(encoder, 7, &nkv);
            set_scalar(encoder, 8, &head_dim);
            set_scalar(encoder, 9, &eps);
        }
        encoder.dispatch_thread_groups(
            MTLSize::new((nq + nkv) as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    }

    /// Dispatch `fused_qk_norm_rope`: RMSNorm + RoPE for Q and K in one dispatch.
    ///
    /// Eliminates intermediate normalised buffers by writing directly from
    /// qkv_buf to the rope output buffers.
    ///
    /// Dispatch: `[nq + nkv, 1, 1]` threadgroups, `[256, 1, 1]` threads
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_fused_qk_norm_rope(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        q_in: &Buffer,
        q_in_offset: u64,
        k_in: &Buffer,
        k_in_offset: u64,
        q_out: &Buffer,
        k_out: &Buffer,
        q_weight: &Buffer,
        k_weight: &Buffer,
        cos_buf: &Buffer,
        sin_buf: &Buffer,
        nq: u32,
        nkv: u32,
        head_dim: u32,
        eps: f32,
    ) {
        encoder.set_compute_pipeline_state(&self.pipelines.fused_qk_norm_rope);
        encoder.set_buffer(0, Some(q_in), q_in_offset);
        encoder.set_buffer(1, Some(k_in), k_in_offset);
        encoder.set_buffer(2, Some(q_out), 0);
        encoder.set_buffer(3, Some(k_out), 0);
        encoder.set_buffer(4, Some(q_weight), 0);
        encoder.set_buffer(5, Some(k_weight), 0);
        encoder.set_buffer(6, Some(cos_buf), 0);
        encoder.set_buffer(7, Some(sin_buf), 0);
        unsafe {
            set_scalar(encoder, 8, &nq);
            set_scalar(encoder, 9, &nkv);
            set_scalar(encoder, 10, &head_dim);
            set_scalar(encoder, 11, &eps);
        }
        encoder.dispatch_thread_groups(
            MTLSize::new((nq + nkv) as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    }

    /// Dispatch `fused_kv_store`: store both K and V into the cache in one dispatch.
    ///
    /// Replaces two separate `kv_cache_store` dispatches.
    ///
    /// `layer_offset` is the `u64` element offset produced by
    /// [`GpuKvCache::layer_offset_elements`](crate::gpu_backend::metal_full_layer::GpuKvCache::layer_offset_elements)
    /// (MET-07). The MSL parameter is still `constant uint&`, so the scalar is
    /// bound as 8 bytes of which the kernel reads the low 4 — exact for every
    /// geometry `GpuKvCache::allocate` admits, because that guard rejects any
    /// cache whose total element count leaves the 32-bit range
    /// (`metal_full_layer::types::KV_CACHE_MAX_ELEMENTS`). B2-15 widens the MSL
    /// binding to `constant ulong&`, after which no cap is needed and this
    /// binding needs no change.
    ///
    /// Dispatch: `[ceil(head_dim/64), nkv, 1]` threadgroups, `[64, 1, 1]` threads
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_fused_kv_store(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        k_data: &Buffer,
        v_data: &Buffer,
        v_data_offset: u64,
        k_cache: &Buffer,
        v_cache: &Buffer,
        nkv: u32,
        head_dim: u32,
        max_seq: u32,
        pos: u32,
        layer_offset: u64,
    ) {
        encoder.set_compute_pipeline_state(&self.pipelines.fused_kv_store);
        encoder.set_buffer(0, Some(k_data), 0);
        encoder.set_buffer(1, Some(v_data), v_data_offset);
        encoder.set_buffer(2, Some(k_cache), 0);
        encoder.set_buffer(3, Some(v_cache), 0);
        unsafe {
            set_scalar(encoder, 4, &head_dim);
            set_scalar(encoder, 5, &nkv);
            set_scalar(encoder, 6, &max_seq);
            set_scalar(encoder, 7, &pos);
            set_scalar(encoder, 8, &layer_offset);
        }
        let tg_x = div_ceil(head_dim as usize, 64) as u64;
        encoder.dispatch_thread_groups(MTLSize::new(tg_x, nkv as u64, 1), MTLSize::new(64, 1, 1));
    }

    /// Dispatch `argmax` — finds the index of the maximum value in a float array.
    ///
    /// Uses a single threadgroup of 1024 threads with shared-memory
    /// tree reduction. Sufficient for vocab ≤ ~500K.
    ///
    /// Buffer layout:
    /// - buffer(0) = data   (f32, input values)
    /// - buffer(1) = result (uint32, single-element output)
    /// - buffer(2) = count  (uint32, scalar)
    ///
    /// Dispatch: `[1, 1, 1]` threadgroups, `[1024, 1, 1]` threads
    pub(crate) fn dispatch_argmax(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        data: &Buffer,
        result: &Buffer,
        count: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.pipelines.argmax);
        encoder.set_buffer(0, Some(data), 0);
        encoder.set_buffer(1, Some(result), 0);
        unsafe {
            set_scalar(encoder, 2, &count);
        }
        // Single threadgroup — 1024 threads cooperate to find max
        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(1024, 1, 1));
    }

    /// Dispatch `topk_f32` — the `k` highest `(id, value)` pairs of a float
    /// array, sorted descending, first-index tie-break (`perf-11`; see the
    /// kernel doc in `kernel_sources/utility.rs` for the full algorithm and
    /// why the cap below is safe rather than a silent truncation).
    ///
    /// This dispatcher only encodes the kernel invocation — it does not
    /// download or interpret `out_ids`/`out_vals`, matching every other
    /// `dispatch_*` method in this file. Wiring a sampled decode request
    /// through this (upload logits → dispatch → download `k` pairs instead
    /// of the full row) is METAL-CONCURRENCY's (wave 4), per the
    /// `topk_f32` field doc on `MetalPipelines`.
    ///
    /// **Before wiring that consumer, read the `-INFINITY` contract** on the
    /// `MSL_TOPK_F32` doc comment (`kernel_sources/utility.rs`): a real,
    /// grammar-masked `-INFINITY` logit and an exhausted pad slot are
    /// indistinguishable in `out_vals`/`out_ids` BY DESIGN — the consumer
    /// must treat `out_vals[i] == -INFINITY` as "no candidate" and ignore
    /// the paired `out_ids[i]`, never sample or rank on it.
    ///
    /// Buffer layout:
    /// - buffer(0) = data     (f32, input values)
    /// - buffer(1) = out_ids  (uint32, `k` winning indices, descending by value)
    /// - buffer(2) = out_vals (f32, `k` winning values, same order)
    /// - buffer(3) = count    (uint32, scalar — length of `data`)
    /// - buffer(4) = k        (uint32, scalar — number of winners requested)
    ///
    /// Dispatch: `[1, 1, 1]` threadgroups, `[1024, 1, 1]` threads — identical
    /// geometry to `dispatch_argmax`.
    ///
    /// # Errors
    ///
    /// Returns [`MetalGraphError::InvalidDimensions`] if `k` exceeds
    /// [`MAX_TOPK_F32`], rather than silently clamping the caller's request
    /// down to fewer winners than asked for. `out_ids`/`out_vals` must each
    /// be sized for at least `k` elements — same convention as every other
    /// `dispatch_*` buffer-sizing precondition in this file.
    ///
    /// The engine-side caller is METAL-CONCURRENCY's (wave 4), so nothing in
    /// the non-test build calls this yet — this file's own
    /// `topk_f32_*`/`dispatch_topk_f32_*` tests do, dispatching the real
    /// kernel and checking its output against a CPU oracle, hence
    /// `#[allow(dead_code)]` (matching `MetalPipelines::
    /// gemm_tq2_g128_v8_tiled`'s precedent for "compiled and tested, not yet
    /// wired into the shipping call path").
    #[allow(dead_code)]
    pub(crate) fn dispatch_topk_f32(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        data: &Buffer,
        out_ids: &Buffer,
        out_vals: &Buffer,
        count: u32,
        k: u32,
    ) -> Result<(), MetalGraphError> {
        if k > MAX_TOPK_F32 {
            return Err(MetalGraphError::InvalidDimensions(format!(
                "dispatch_topk_f32: k={k} exceeds MAX_TOPK_F32={MAX_TOPK_F32}"
            )));
        }
        encoder.set_compute_pipeline_state(&self.pipelines.topk_f32);
        encoder.set_buffer(0, Some(data), 0);
        encoder.set_buffer(1, Some(out_ids), 0);
        encoder.set_buffer(2, Some(out_vals), 0);
        unsafe {
            set_scalar(encoder, 3, &count);
            set_scalar(encoder, 4, &k);
        }
        // Single threadgroup — 1024 threads cooperate, exactly as `argmax`.
        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(1024, 1, 1));
        Ok(())
    }

    /// Resolve (and cache) a compute pipeline for `name` from the shared
    /// embedded metallib.
    ///
    /// Convenience wrapper around [`MetalPipelines::pipeline_for`] for a
    /// caller that already holds a `&MetalGraph` (e.g. via
    /// `MetalGraph::global()`) and so has both the library and the device it
    /// needs without threading the device through separately.
    ///
    /// MET-10: the K-quant, standard-quant, FP8 GEMV and FP8 batch-prefill
    /// families resolve every pipeline through this —
    /// `graph.pipeline_for("gemv_q4k")` (etc.) — reusing the session's
    /// device and command queue and the embedded/disk-cached combined
    /// metallib, instead of compiling a private library from source per
    /// kernel on first use as they used to.
    pub(crate) fn pipeline_for(&self, name: &str) -> Result<ComputePipelineState, MetalGraphError> {
        self.pipelines.pipeline_for(&self.device, name)
    }

    // ─────────────────────────────────────────────────────────────────────
    // Batch-prefill dispatch helpers (GEMM, batched SwiGLU, batched RMSNorm)
    // ─────────────────────────────────────────────────────────────────────

    /// Dispatch batched SwiGLU for `batch_size` vectors.
    ///
    /// Input: `gate_up[b * inter * 2 .. b * inter * 2 + inter * 2]` for each batch `b`.
    /// Output: `output[b * inter .. b * inter + inter]`.
    ///
    /// Buffer layout:
    /// - buffer(0) = gate_up    (f32, `batch_size × inter × 2`)
    /// - buffer(1) = output     (f32, `batch_size × inter`)
    /// - buffer(2) = inter      (u32)
    /// - buffer(3) = batch_size (u32)
    ///
    /// Dispatch: `[ceil(inter / 256), batch_size, 1]` threadgroups, `[256, 1, 1]` threads
    #[allow(dead_code)]
    pub(crate) fn dispatch_batched_swiglu(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        gate_up: &Buffer,
        output: &Buffer,
        inter: u32,
        batch_size: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.pipelines.batched_swiglu);
        encoder.set_buffer(0, Some(gate_up), 0);
        encoder.set_buffer(1, Some(output), 0);
        unsafe {
            set_scalar(encoder, 2, &inter);
            set_scalar(encoder, 3, &batch_size);
        }

        let tg_x = div_ceil(inter as usize, 256) as u64;
        encoder.dispatch_thread_groups(
            MTLSize::new(tg_x, batch_size as u64, 1),
            MTLSize::new(256, 1, 1),
        );
    }

    /// Dispatch single-vector SwiGLU using the `batched_swiglu` pipeline with `batch_size=1`.
    ///
    /// Thin convenience wrapper for the ternary decode path: the TQ2 GEMV produces a
    /// `2 × inter` gate-up buffer, after which `silu(gate) * up` is applied element-wise
    /// to yield the `inter`-wide FFN activation. Mirrors the Q1 fused `fused_gate_up_swiglu_q1`
    /// kernel's post-projection behaviour but as a separate dispatch (since ternary lacks
    /// a fused variant).
    ///
    /// Buffer layout:
    /// - buffer(0) = gate_up_buf (f32, `2 × inter`, gate in `[0, inter)`, up in `[inter, 2·inter)`)
    /// - buffer(1) = output_buf  (f32, `inter`, receives `silu(gate) * up`)
    /// - buffer(2) = inter       (u32, set_bytes)
    /// - buffer(3) = batch_size  (u32, set_bytes, always `1`)
    ///
    /// Dispatch: `[ceil(inter / 256), 1, 1]` threadgroups, `[256, 1, 1]` threads.
    #[allow(dead_code)]
    pub(crate) fn dispatch_swiglu_single(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        gate_up_buf: &Buffer,
        output_buf: &Buffer,
        inter: u32,
    ) {
        self.dispatch_batched_swiglu(encoder, gate_up_buf, output_buf, inter, 1);
    }

    /// Dispatch batched RMSNorm for `batch_size` position vectors.
    ///
    /// Uses the existing `batched_rmsnorm_v2` kernel which handles multiple
    /// vectors via `threadgroup_position_in_grid`.
    ///
    /// Input: `batch_size` vectors of `dim` floats, contiguous (`input[b * dim + i]`).
    /// Weight: single weight vector of `dim` floats (shared across all positions).
    /// Output: `batch_size` normalised vectors of `dim` floats.
    ///
    /// Dispatch: `[batch_size, 1, 1]` threadgroups, `[256, 1, 1]` threads
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_batched_rmsnorm(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        input: &Buffer,
        weight: &Buffer,
        output: &Buffer,
        eps: f32,
        dim: u32,
        batch_size: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.pipelines.batched_rmsnorm_v2);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(weight), 0);
        encoder.set_buffer(2, Some(output), 0);
        unsafe {
            set_scalar(encoder, 3, &eps);
            set_scalar(encoder, 4, &dim);
        }

        // One threadgroup per position in the batch
        encoder.dispatch_thread_groups(
            MTLSize::new(batch_size as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    }

    /// Dispatch batched attention scores V2: 128-thread TGs with position batching.
    /// Each TG processes `batch_stride` positions instead of 1, reducing TG scheduling overhead.
    ///
    /// # Preconditions
    ///
    /// * `head_dim <= ATTENTION_SCORES_V2_MAX_HEAD_DIM` — see that constant.
    ///   This is the single entry point shared by decode
    ///   (`metal_full_layer::functions_2`) and prefill (`metal_prefill::functions`),
    ///   so the check lives here rather than at each call site.
    /// * `cache_layer_offset` is the `u64` element offset from
    ///   `GpuKvCache::layer_offset_elements`; see
    ///   [`Self::dispatch_fused_kv_store`] for why binding it against the
    ///   kernel's `constant uint&` is exact for every admitted geometry.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_attention_scores_v2(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        queries: &Buffer,
        k_cache: &Buffer,
        scores: &Buffer,
        head_dim: u32,
        n_q: u32,
        n_kv: u32,
        heads_per_group: u32,
        max_seq: u32,
        seq_len: u32,
        inv_sqrt_hd: f32,
        cache_layer_offset: u64,
    ) {
        debug_assert!(
            head_dim <= ATTENTION_SCORES_V2_MAX_HEAD_DIM,
            "batched_attention_scores_v2 stages only {ATTENTION_SCORES_V2_MAX_HEAD_DIM} query \
             dims; head_dim {head_dim} would silently score on a partial dot product"
        );
        if head_dim > ATTENTION_SCORES_V2_MAX_HEAD_DIM {
            tracing::error!(
                head_dim,
                max_head_dim = ATTENTION_SCORES_V2_MAX_HEAD_DIM,
                "batched_attention_scores_v2 staging array is too small for this head_dim; \
                 scores will be computed over the first {ATTENTION_SCORES_V2_MAX_HEAD_DIM} dims \
                 only (MET-01 — kernel widening lands with B2-15)"
            );
        }
        let batch_stride: u32 = 16; // Process 16 positions per TG
        encoder.set_compute_pipeline_state(&self.pipelines.batched_attention_scores_v2);
        encoder.set_buffer(0, Some(queries), 0);
        encoder.set_buffer(1, Some(k_cache), 0);
        encoder.set_buffer(2, Some(scores), 0);
        unsafe {
            set_scalar(encoder, 3, &head_dim);
            set_scalar(encoder, 4, &n_q);
            set_scalar(encoder, 5, &n_kv);
            set_scalar(encoder, 6, &heads_per_group);
            set_scalar(encoder, 7, &max_seq);
            set_scalar(encoder, 8, &seq_len);
            set_scalar(encoder, 9, &inv_sqrt_hd);
            set_scalar(encoder, 10, &cache_layer_offset);
            set_scalar(encoder, 11, &batch_stride);
        }
        let tg_y = div_ceil(seq_len as usize, batch_stride as usize);
        encoder.dispatch_thread_groups(
            MTLSize::new(n_q as u64, tg_y as u64, 1),
            MTLSize::new(128, 1, 1),
        );
    }

    // ─────────────────────────────────────────────────────────────────────
    // DiT joint attention (flash-attention simdgroup_matrix — shipping path)
    // ─────────────────────────────────────────────────────────────────────

    /// Dispatch `joint_attention_flash_f32` — the flash-attention (online-softmax)
    /// `simdgroup_float8x8` HW-matrix DiT joint (txt+img) multi-head attention
    /// (non-causal, f32 accumulate, head→token transpose folded into the store).
    ///
    /// Buffer layout:
    /// - buffer(0) = q         (f32, head-major `[num_heads × seq × head_dim]`)
    /// - buffer(1) = k         (f32, head-major `[num_heads × seq × head_dim]`)
    /// - buffer(2) = v         (f32, head-major `[num_heads × seq × head_dim]`)
    /// - buffer(3) = out       (f32, token-major `[seq × (num_heads*head_dim)]`)
    /// - buffer(4) = num_heads (u32, set_bytes)
    /// - buffer(5) = seq       (u32, set_bytes)
    /// - buffer(6) = head_dim  (u32, set_bytes)
    /// - buffer(7) = scale     (f32, set_bytes — `1/sqrt(head_dim)`)
    ///
    /// One threadgroup computes a whole **query-tile** of `FA_BQ` (= 64) output
    /// rows for a head, driving the hardware 8×8 matrix units for both `Q·Kᵀ` and
    /// `P·V`. The grid is `[ceil(seq/FA_BQ), num_heads, 1]` and each threadgroup
    /// runs `FA_SIMDGROUPS·32` (= 256) threads (8 simdgroups).
    ///
    /// The `FA_BQ` / `FA_BK` tile constants and the 8-simdgroup (256-thread)
    /// shape here MUST match the `joint_attention_flash_f32` MSL kernel
    /// (`DIT_FLASH_BQ` / `DIT_FLASH_BK`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_joint_attention_flash(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
        out: &Buffer,
        num_heads: u32,
        seq: u32,
        head_dim: u32,
        scale: f32,
    ) {
        use crate::gpu_backend::kernel_sources::DIT_FLASH_BQ;
        const SIMDGROUPS: u64 = 8; // -> 256 threads (must match FA_SIMDGROUPS in the MSL)
        const THREADS: u64 = SIMDGROUPS * 32;

        encoder.set_compute_pipeline_state(&self.pipelines.joint_attention_flash_f32);
        encoder.set_buffer(0, Some(q), 0);
        encoder.set_buffer(1, Some(k), 0);
        encoder.set_buffer(2, Some(v), 0);
        encoder.set_buffer(3, Some(out), 0);
        unsafe {
            set_scalar(encoder, 4, &num_heads);
            set_scalar(encoder, 5, &seq);
            set_scalar(encoder, 6, &head_dim);
            set_scalar(encoder, 7, &scale);
        }

        // One threadgroup per (query-tile of FA_BQ rows, head); 256 threads each.
        let tg_x = div_ceil(seq as usize, DIT_FLASH_BQ) as u64;
        encoder.dispatch_thread_groups(
            MTLSize::new(tg_x, num_heads as u64, 1),
            MTLSize::new(THREADS, 1, 1),
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests — u64 KV layer offset binding (MET-07)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(all(test, feature = "metal", target_os = "macos"))]
mod tests {
    use super::*;
    use crate::gpu_backend::metal_full_layer::types::kv_layer_offset_elements;
    use metal::{Device, MTLResourceOptions};

    /// The Rust side now hands `fused_kv_store` a `u64` layer offset while the
    /// MSL parameter is still `constant uint&`. Prove on the device that the
    /// store lands at the intended address for a *non-zero* layer — i.e. that
    /// the 8-byte `set_bytes` against a 4-byte kernel parameter is accepted and
    /// read as the low word.
    #[test]
    fn fused_kv_store_honours_a_u64_layer_offset() {
        if Device::system_default().is_none() {
            return;
        }
        let graph = MetalGraph::global().expect("MetalGraph::global");

        let (n_layers, nkv, max_seq, head_dim) = (4usize, 2usize, 8usize, 16usize);
        let layer_idx = 3usize;
        let pos = 5usize;
        let total = n_layers * nkv * max_seq * head_dim;

        let shared = MTLResourceOptions::StorageModeShared;
        let k_data: Vec<f32> = (0..nkv * head_dim).map(|i| i as f32 + 1.0).collect();
        let v_data: Vec<f32> = (0..nkv * head_dim).map(|i| -(i as f32) - 1.0).collect();
        let k_buf = graph.device.new_buffer((k_data.len() * 4) as u64, shared);
        let v_buf = graph.device.new_buffer((v_data.len() * 4) as u64, shared);
        // SAFETY: both buffers are StorageModeShared and sized for the slices.
        unsafe {
            super::super::metal_graph::upload_f32(&k_buf, &k_data);
            super::super::metal_graph::upload_f32(&v_buf, &v_data);
        }
        let k_cache = graph.device.new_buffer((total * 2) as u64, shared);
        let v_cache = graph.device.new_buffer((total * 2) as u64, shared);
        // REQUIRED #5 (wave-1+1.5 gatekeeper review): Metal does NOT
        // zero-initialise newly allocated buffers — `newBufferWithLength:
        // options:` makes no such guarantee, so the "untouched" assertions
        // below need an explicit known-zero baseline instead of relying on
        // allocator/driver behaviour that merely happened to read as zero.
        // SAFETY: both buffers are freshly allocated, `StorageModeShared`
        // (CPU- and GPU-visible with no synchronization needed before the
        // command buffer that writes them runs), and `total * 2` is exactly
        // their allocated byte length (`total` `half::f16` elements each).
        unsafe {
            std::ptr::write_bytes(k_cache.contents() as *mut u8, 0u8, total * 2);
            std::ptr::write_bytes(v_cache.contents() as *mut u8, 0u8, total * 2);
        }

        let layer_offset = kv_layer_offset_elements(layer_idx, nkv, max_seq, head_dim);
        assert_eq!(layer_offset, (3 * 2 * 8 * 16) as u64);

        let cmd = graph.command_queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        graph.dispatch_fused_kv_store(
            encoder,
            &k_buf,
            &v_buf,
            0,
            &k_cache,
            &v_cache,
            nkv as u32,
            head_dim as u32,
            max_seq as u32,
            pos as u32,
            layer_offset,
        );
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let k_ptr = k_cache.contents() as *const half::f16;
        let v_ptr = v_cache.contents() as *const half::f16;
        for head in 0..nkv {
            for d in 0..head_dim {
                let dst = layer_offset as usize + (head * max_seq + pos) * head_dim + d;
                let src = head * head_dim + d;
                // SAFETY: `dst < total` and the buffer is shared and initialised.
                let got_k = unsafe { std::ptr::read(k_ptr.add(dst)) }.to_f32();
                let got_v = unsafe { std::ptr::read(v_ptr.add(dst)) }.to_f32();
                assert_eq!(got_k, k_data[src], "K at head {head} dim {d}");
                assert_eq!(got_v, v_data[src], "V at head {head} dim {d}");
            }
        }
        // Nothing outside the addressed slab was touched.
        let first_of_layer = layer_offset as usize;
        for idx in [0usize, first_of_layer - 1, first_of_layer + head_dim - 1] {
            // SAFETY: indices are inside the allocated cache.
            let stray = unsafe { std::ptr::read(k_ptr.add(idx)) }.to_f32();
            assert_eq!(stray, 0.0, "cache element {idx} must be untouched");
        }
    }

    /// The staging-array cap that `batched_attention_scores_v2` relies on must
    /// stay in sync with the MSL text until B2-15 widens it.
    #[test]
    fn attention_scores_v2_head_dim_cap_matches_the_msl_staging_array() {
        use crate::gpu_backend::kernel_sources::MSL_BATCHED_ATTENTION_SCORES_V2;
        assert!(
            MSL_BATCHED_ATTENTION_SCORES_V2.contains(&format!(
                "threadgroup float shared_q[{ATTENTION_SCORES_V2_MAX_HEAD_DIM}]"
            )),
            "MSL staging array size changed; update ATTENTION_SCORES_V2_MAX_HEAD_DIM"
        );
    }

    // ─────────────────────────────────────────────────────────────────────
    // `topk_f32` (perf-11 sampled-decode partial reduction)
    // ─────────────────────────────────────────────────────────────────────

    /// Deterministic xorshift64* — failures are reproducible from the seed
    /// alone (same construction as `tests/gpu_argmax_tiebreak.rs`).
    struct TopkRng(u64);

    impl TopkRng {
        fn new(seed: u64) -> Self {
            Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
        }

        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }

        fn f32_range(&mut self, lo: f32, hi: f32) -> f32 {
            let unit = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
            lo + unit * (hi - lo)
        }
    }

    /// CPU oracle for `topk_f32`: repeatedly select the arg-max over the
    /// elements not yet chosen, using the same first-index tie-break rule as
    /// the kernel (and `argmax`/`argmax_first`) — an exact value tie is won
    /// by the smaller original index. Pads with `(0, -INFINITY)` once every
    /// real element has been picked, mirroring the kernel's own `k > count`
    /// padding.
    fn cpu_topk_reference(data: &[f32], k: usize) -> (Vec<u32>, Vec<f32>) {
        let mut remaining: Vec<u32> = (0..data.len() as u32).collect();
        let mut ids = Vec::with_capacity(k);
        let mut vals = Vec::with_capacity(k);
        for _ in 0..k {
            if remaining.is_empty() {
                ids.push(0);
                vals.push(f32::NEG_INFINITY);
                continue;
            }
            let mut best_pos = 0usize;
            for p in 1..remaining.len() {
                let best_idx = remaining[best_pos];
                let cand_idx = remaining[p];
                let (bv, cv) = (data[best_idx as usize], data[cand_idx as usize]);
                if cv > bv || (cv == bv && cand_idx < best_idx) {
                    best_pos = p;
                }
            }
            let idx = remaining.remove(best_pos);
            ids.push(idx);
            vals.push(data[idx as usize]);
        }
        (ids, vals)
    }

    /// Round-trip `data` through the real `topk_f32` GPU kernel and return
    /// the `k` `(id, value)` pairs it wrote.
    fn gpu_topk(graph: &MetalGraph, data: &[f32], k: u32) -> (Vec<u32>, Vec<f32>) {
        let shared = MTLResourceOptions::StorageModeShared;
        let data_buf = graph.device.new_buffer((data.len() * 4) as u64, shared);
        // SAFETY: freshly allocated, `StorageModeShared`, sized for `data`.
        unsafe {
            super::super::metal_graph::upload_f32(&data_buf, data);
        }
        let slot_count = k.max(1) as u64;
        let ids_buf = graph.device.new_buffer(slot_count * 4, shared);
        let vals_buf = graph.device.new_buffer(slot_count * 4, shared);

        let cmd = graph.command_queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        graph
            .dispatch_topk_f32(
                encoder,
                &data_buf,
                &ids_buf,
                &vals_buf,
                data.len() as u32,
                k,
            )
            .expect("dispatch_topk_f32 must accept k <= MAX_TOPK_F32 in this test");
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let ids_ptr = ids_buf.contents() as *const u32;
        let vals_ptr = vals_buf.contents() as *const f32;
        let mut ids = Vec::with_capacity(k as usize);
        let mut vals = Vec::with_capacity(k as usize);
        for i in 0..k as usize {
            // SAFETY: `ids_buf`/`vals_buf` hold exactly `k` elements each and
            // the command buffer above has completed.
            unsafe {
                ids.push(std::ptr::read(ids_ptr.add(i)));
                vals.push(std::ptr::read(vals_ptr.add(i)));
            }
        }
        (ids, vals)
    }

    /// The GPU kernel must agree with the CPU oracle across a spread of
    /// vocab sizes (including one that is not a multiple of the 1024-thread
    /// threadgroup) and `k` values, on data with no exact ties — so this
    /// pins the core reduction, independent of the tie-break rule (covered
    /// separately below).
    #[test]
    fn topk_f32_matches_cpu_reference_on_random_data() {
        if Device::system_default().is_none() {
            return;
        }
        let graph = MetalGraph::global().expect("MetalGraph::global");

        for &n in &[500usize, 4096] {
            for &k in &[1u32, 5, 20] {
                for seed in 0u64..3 {
                    let mut rng = TopkRng::new(n as u64 * 1_000_003 + k as u64 * 97 + seed + 1);
                    let data: Vec<f32> = (0..n).map(|_| rng.f32_range(-100.0, 100.0)).collect();

                    let (got_ids, got_vals) = gpu_topk(&graph, &data, k);
                    let (want_ids, want_vals) = cpu_topk_reference(&data, k as usize);

                    assert_eq!(got_ids, want_ids, "n={n} k={k} seed={seed}: id mismatch");
                    for (g, w) in got_vals.iter().zip(want_vals.iter()) {
                        assert!(
                            (g - w).abs() < 1e-4,
                            "n={n} k={k} seed={seed}: value mismatch {g} vs {w}"
                        );
                    }
                }
            }
        }
    }

    /// Bonsai 2's real vocabulary size (248 320, not a multiple of the
    /// 1024-thread threadgroup) at the recommended sampling `top_k = 20`.
    #[test]
    fn topk_f32_handles_the_real_bonsai2_vocab_size() {
        if Device::system_default().is_none() {
            return;
        }
        let graph = MetalGraph::global().expect("MetalGraph::global");

        let n = 248_320usize;
        let k = 20u32;
        let mut rng = TopkRng::new(0x0B0A_5A12);
        let data: Vec<f32> = (0..n).map(|_| rng.f32_range(-50.0, 50.0)).collect();

        let (got_ids, got_vals) = gpu_topk(&graph, &data, k);
        let (want_ids, want_vals) = cpu_topk_reference(&data, k as usize);
        assert_eq!(got_ids, want_ids);
        for (g, w) in got_vals.iter().zip(want_vals.iter()) {
            assert!((g - w).abs() < 1e-4, "{g} vs {w}");
        }
    }

    /// Multiple exact ties at the maximum: the smaller original index must
    /// win every one of the tied slots, in ascending order — the same
    /// `argmax_first` rule `gpu_argmax_tiebreak.rs` pins for plain `argmax`,
    /// exercised here across `k` winners instead of one.
    #[test]
    fn topk_f32_breaks_ties_by_smallest_index_ascending() {
        if Device::system_default().is_none() {
            return;
        }
        let graph = MetalGraph::global().expect("MetalGraph::global");

        const TIE_VALUE: f32 = 42.0;
        let tied_indices = [50u32, 800, 1200, 1500, 1900];
        let n = 2000usize;
        let mut rng = TopkRng::new(0x7113);
        let mut data: Vec<f32> = (0..n).map(|_| rng.f32_range(-8.0, 8.0)).collect();
        for &idx in &tied_indices {
            data[idx as usize] = TIE_VALUE;
        }

        for &k in &[1u32, 3, 5] {
            let (got_ids, got_vals) = gpu_topk(&graph, &data, k);
            let expected_ids: Vec<u32> = tied_indices[..k as usize].to_vec();
            assert_eq!(
                got_ids, expected_ids,
                "k={k}: must return ties ascending-index-first"
            );
            assert!(
                got_vals.iter().all(|&v| v == TIE_VALUE),
                "k={k}: every returned value must be the tied maximum"
            );
        }
    }

    /// `k > count`: once every real element is exhausted, the remaining
    /// slots must be padded with `(0, -INFINITY)`, not a repeated winner or
    /// an out-of-bounds read.
    #[test]
    fn topk_f32_pads_when_k_exceeds_count() {
        if Device::system_default().is_none() {
            return;
        }
        let graph = MetalGraph::global().expect("MetalGraph::global");

        let data = [3.0f32, 1.0, 4.0, 1.5, 0.5];
        let k = 8u32;
        let (got_ids, got_vals) = gpu_topk(&graph, &data, k);
        let (want_ids, want_vals) = cpu_topk_reference(&data, k as usize);

        assert_eq!(got_ids, want_ids);
        for (i, (&g, &w)) in got_vals.iter().zip(want_vals.iter()).enumerate() {
            if i < data.len() {
                assert!((g - w).abs() < 1e-4, "slot {i}: {g} vs {w}");
                assert!(g.is_finite(), "slot {i}: real element must be finite");
            } else {
                assert!(
                    g.is_infinite() && g.is_sign_negative(),
                    "padding slot {i} must be -INFINITY, got {g}"
                );
                assert_eq!(got_ids[i], 0, "padding slot {i} must carry id 0");
            }
        }
    }

    /// A `k` above [`MAX_TOPK_F32`] must be a typed, catchable error — never
    /// a silently truncated result and never an out-of-bounds `picked[256]`
    /// write in the kernel.
    #[test]
    fn dispatch_topk_f32_rejects_k_above_the_documented_cap() {
        if Device::system_default().is_none() {
            return;
        }
        let graph = MetalGraph::global().expect("MetalGraph::global");

        let shared = MTLResourceOptions::StorageModeShared;
        let data_buf = graph.device.new_buffer(4 * 4, shared);
        let ids_buf = graph.device.new_buffer(4, shared);
        let vals_buf = graph.device.new_buffer(4, shared);

        let cmd = graph.command_queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        let result =
            graph.dispatch_topk_f32(encoder, &data_buf, &ids_buf, &vals_buf, 4, MAX_TOPK_F32 + 1);
        encoder.end_encoding();

        match result {
            Err(MetalGraphError::InvalidDimensions(msg)) => {
                assert!(msg.contains("MAX_TOPK_F32"), "{msg}");
            }
            other => panic!("expected InvalidDimensions, got {other:?}"),
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // `pipeline_for` (MET-10 escape hatch)
    // ─────────────────────────────────────────────────────────────────────

    /// End-to-end proof that [`MetalGraph::pipeline_for`] (MET-10's escape
    /// hatch) resolves a *working*, *correct* pipeline for a kernel that
    /// rides the combined metallib only through this on-demand lookup —
    /// `gemv_q4k` is never extracted into a named `MetalPipelines` field.
    /// Dispatches it entirely by name and checks the output against the
    /// crate's own scalar `gemv_q4k` CPU reference.
    ///
    /// This doubles as a regression guard for the `kq_scale_min_k4` ->
    /// `kq_scale_min_k4_q4k`/`_q5k` rename this package made in
    /// `kernel_sources/k_quant.rs`: both kernels used to define
    /// `kq_scale_min_k4` identically, which was legal only because each
    /// compiled into its own separate `MTLLibrary`; concatenated into one
    /// combined library (as `build_combined_msl` now does for both) an
    /// unrenamed duplicate would be an MSL redefinition error, so a passing
    /// dispatch here proves the combined library actually compiled AND that
    /// the rename did not perturb `gemv_q4k`'s numerics.
    #[test]
    fn pipeline_for_resolves_and_dispatches_a_kquant_kernel_by_name() {
        if Device::system_default().is_none() {
            return;
        }
        let graph = MetalGraph::global().expect("MetalGraph::global");

        // One Q4_K super-block (256 weights). Every byte pattern is a valid
        // Q4_K block (no reserved codes to avoid, unlike ternary), so a
        // simple deterministic fill exercises the real decode path.
        let mut scales = [0u8; 12];
        let mut qs = [0u8; 128];
        for (i, s) in scales.iter_mut().enumerate() {
            *s = 3u8.wrapping_add((i as u8).wrapping_mul(7));
        }
        for (i, q) in qs.iter_mut().enumerate() {
            *q = 5u8.wrapping_add((i as u8).wrapping_mul(13));
        }
        let block = oxibonsai_core::BlockQ4K {
            d: half::f16::from_f32(0.073),
            dmin: half::f16::from_f32(0.019),
            scales,
            qs,
        };
        let k = 256usize;
        let input: Vec<f32> = (0..k).map(|i| (i as f32) * 0.01 - 1.28).collect();

        let mut expected = [0f32; 1];
        crate::gemv_q4k::gemv_q4k(std::slice::from_ref(&block), &input, &mut expected, 1, k)
            .expect("scalar gemv_q4k reference");

        let shared = MTLResourceOptions::StorageModeShared;
        // SAFETY: `block` is a valid, fully initialised `#[repr(C)]`
        // `BlockQ4K`, and `size_of::<BlockQ4K>()` is exactly the kernel's
        // documented 144-byte stride.
        let block_bytes = unsafe {
            std::slice::from_raw_parts(
                std::ptr::from_ref(&block).cast::<u8>(),
                std::mem::size_of::<oxibonsai_core::BlockQ4K>(),
            )
        };
        let block_buf = graph.device.new_buffer_with_data(
            block_bytes.as_ptr() as *const std::ffi::c_void,
            block_bytes.len() as u64,
            shared,
        );
        let input_buf = graph.device.new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(input.as_slice()) as u64,
            shared,
        );
        let output_buf = graph.device.new_buffer(4, shared);
        // SAFETY: freshly allocated, `StorageModeShared`, exactly the 4
        // bytes of the one `f32` this dispatch writes.
        unsafe {
            std::ptr::write_bytes(output_buf.contents() as *mut u8, 0u8, 4);
        }

        let pso = graph
            .pipeline_for("gemv_q4k")
            .expect("pipeline_for must resolve gemv_q4k from the combined metallib");
        // A second lookup must hit the cache and still resolve.
        graph
            .pipeline_for("gemv_q4k")
            .expect("pipeline_for must be idempotent on a cache hit");

        let n_rows: u32 = 1;
        let k_u32 = k as u32;
        let cmd = graph.command_queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pso);
        encoder.set_buffer(0, Some(&block_buf), 0);
        encoder.set_buffer(1, Some(&input_buf), 0);
        encoder.set_buffer(2, Some(&output_buf), 0);
        // SAFETY: `encoder` is active with buffers 0-2 already bound above.
        unsafe {
            set_scalar(encoder, 3, &n_rows);
            set_scalar(encoder, 4, &k_u32);
        }
        // Per the kernel's own doc: grid `[ceil(n_rows/8), 1, 1]`, block
        // `[256, 1, 1]` (8 simdgroups x 32 lanes, one row per simdgroup).
        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let got = unsafe { std::ptr::read(output_buf.contents() as *const f32) };
        assert!(
            (got - expected[0]).abs() < 1e-2,
            "gemv_q4k via pipeline_for: got {got}, scalar reference {}",
            expected[0]
        );
        assert!(
            got.abs() > 1e-6,
            "degenerate fixture: output must be non-zero"
        );

        // An unknown name must be a typed, catchable error.
        match graph.pipeline_for("not_a_real_kernel_name") {
            Err(MetalGraphError::EncodingFailed(_)) => {}
            other => panic!("expected EncodingFailed for an unknown name, got {other:?}"),
        }
    }

    /// MET-10 first-use latency: the K-quant / Q-std / FP8 families used to
    /// call `device.new_library_with_source(...)` once per kernel, uncached,
    /// on first use of that kernel — the finding cites ~0.3-1s per library
    /// for this class of call. They now resolve through `pipeline_for`
    /// (`metal_graph/tests_no_private_library.rs` is the guard that keeps it
    /// that way); "before" below reproduces the retired call shape
    /// for comparison only. `pipeline_for` resolves the same kernel against
    /// the combined
    /// embedded/disk-cached metallib the shared `MetalDevice` already loaded
    /// once for every other kernel family, so once that device exists, a
    /// first `pipeline_for` call for a K-quant/Q-std/FP8 kernel is a
    /// `get_function` + pipeline-state creation against an *already-loaded*
    /// library, not a fresh `MTLLibrary` compile from source text — that
    /// structural difference (no separate compile call at all) holds
    /// regardless of the exact timing below.
    ///
    /// **On the numbers this prints**: macOS's Metal stack keeps its own
    /// shader-compilation cache (compiled AIR/binary keyed by source),
    /// persisted outside this process. By the time this test runs, this
    /// exact `MSL_GEMV_Q4K_V1` source text has typically already been
    /// compiled many times over in this session alone (every `cargo build`
    /// re-runs `build.rs`'s `xcrun` compile of the combined MSL, which
    /// contains this same text) — so the "before" measurement below is very
    /// likely a warm system-cache hit, not the finder's cited cold-start
    /// cost, and this test cannot reproduce or refute that ~0.3-1s figure.
    /// Treat the printed numbers as "same-process relative cost of the two
    /// call shapes on whatever cache state the host happens to be in", not
    /// as an absolute or cold-start measurement.
    ///
    /// **Deliberately prints, never asserts, on the two durations** (wave-3
    /// re-review): an earlier version of this test ended in
    /// `assert!(after < before)`, a wall-clock comparison between two
    /// `Instant::elapsed()` values on a shared, load-dependent host. Because
    /// this test is `#[ignore]`d it never runs in the gate, but a wall-clock
    /// `assert!` left in an `#[ignore]`d test is still a flake trap for
    /// whoever un-ignores it later — the doc comment above already explains
    /// why `before` can legitimately land either side of `after` depending on
    /// system shader-cache state. Do not restore the assertion; `eprintln!`
    /// the numbers instead, exactly as below.
    ///
    /// `#[ignore]`d: a wall-clock measurement, not a correctness assertion —
    /// same convention this package's wave-2.5 addendum cites for the
    /// KERN-PARALLEL / B2-03 in-crate timing tests. Run explicitly with
    /// `cargo test -p oxibonsai-kernels --features metal \
    /// met_10_first_use_latency_before_vs_after -- --ignored --nocapture`
    /// to reproduce the numbers recorded in this package's notes.
    #[test]
    #[ignore = "wall-clock measurement, not a correctness gate — see doc comment"]
    fn met_10_first_use_latency_before_vs_after() {
        if Device::system_default().is_none() {
            eprintln!("no Metal device; skipping MET-10 latency measurement");
            return;
        }
        let device = Device::system_default().expect("checked is_none() above");

        // BEFORE: `new_library_with_source` on one K-quant kernel's MSL
        // alone — exactly the call shape `metal_k_quant_kernels.rs` used per
        // kernel until MET-10 (see the doc above for why "before" here may be
        // warm rather than truly cold).
        let src = crate::gpu_backend::kernel_sources::MSL_GEMV_Q4K_V1;
        let before_start = std::time::Instant::now();
        let lib = device
            .new_library_with_source(src, &metal::CompileOptions::new())
            .expect("compiling a single K-quant kernel from source must succeed");
        lib.get_function("gemv_q4k", None)
            .expect("gemv_q4k must be the kernel's entry point");
        let before = before_start.elapsed();

        // AFTER: `MetalGraph::global()` first — whatever it costs (a fresh
        // process-wide load, or nothing if another test already triggered
        // it) happens before the timer starts, so `after` measures only the
        // by-name resolution against an *already-resident* library.
        let graph = MetalGraph::global().expect("MetalGraph::global");
        let after_start = std::time::Instant::now();
        graph
            .pipeline_for("gemv_q4k")
            .expect("gemv_q4k must resolve from the combined metallib");
        let after = after_start.elapsed();

        eprintln!(
            "[MET-10] new_library_with_source for one kernel: {before:?}; \
             pipeline_for against the already-loaded combined metallib: {after:?} \
             (see this test's doc comment: 'before' may be a warm system \
             shader-cache hit, not the finder's cited cold-start cost)"
        );
        // No `assert!` here by design — see the doc comment above.
    }
}
