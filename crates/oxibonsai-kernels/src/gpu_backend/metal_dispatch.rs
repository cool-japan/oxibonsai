//! Individual Metal kernel dispatch methods for `MetalGraph`.
//!
//! Each `dispatch_*` method encodes a single GPU kernel invocation
//! into the currently active compute command encoder.

#![cfg(feature = "metal")]

use metal::{Buffer, MTLSize};

use super::metal_graph::{div_ceil, set_scalar, MetalGraph};

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
    /// `batch_size` correctly, unlike `dispatch_gemm_q1_v7` whose kernel
    /// silently caps at 8 columns).
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
        layer_offset: u32,
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
        cache_layer_offset: u32,
    ) {
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
}
