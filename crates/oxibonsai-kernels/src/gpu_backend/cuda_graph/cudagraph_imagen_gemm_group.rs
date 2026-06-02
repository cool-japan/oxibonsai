//! # CudaGraph - imagen GEMM (image-generation / FLUX.2) Methods
//!
//! Host-side encode for the two image-generation CUDA-core GEMMs that mirror the
//! Metal image path's `encode_gemm_f32` / `encode_gemm_tq2`
//! (`metal_graph/graph.rs`):
//!
//! - [`CudaGraph::encode_gemm_f32`]  -> launches the `gemm_f32` kernel.
//! - [`CudaGraph::encode_gemm_tq2`]  -> launches the `gemm_tq2` kernel.
//!
//! Both are FULL-M (`grid.y` tiles + in-kernel `m_local < M` clamp), so they have
//! no cap-of-8 trap. The public signatures match the Metal siblings exactly
//! (`weight, input, output, m, n_rows, k`), differing only in the weight handle
//! type (a device `Arc<CudaSlice<_>>` instead of a `MetalWeightHandle`).
//!
//! The kernel functions (`gemm_f32`, `gemm_tq2`) must be loaded into
//! `CudaModules` from [`CUDA_IMAGEN_GEMM_SRC`](super::super::cuda_imagen_gemm_kernels::CUDA_IMAGEN_GEMM_SRC)
//! by the integration step (`CudaGraph::new`); this module assumes
//! `self.modules.gemm_f32` / `self.modules.gemm_tq2` are present.

#![cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use std::sync::Arc;

use super::types::CudaGraphError;

use super::cudagraph_type::CudaGraph;

impl CudaGraph {
    /// Launch `gemm_f32` on the default stream.
    ///
    /// `pub(crate)` so the VAE conv path
    /// ([`encode_conv2d_f32_im2col`](Self::encode_conv2d_f32_im2col)) can run the
    /// GEMM directly on its device-resident im2col patch buffer (no D2H/H2D
    /// round-trip of the patches).
    ///
    /// # Safety
    /// Caller must ensure all slices are valid device pointers on `self.stream`,
    /// that `d_input` holds ≥ `m*k` f32 and `d_output` ≥ `m*n_rows` f32, and that
    /// `d_weight` holds ≥ `n_rows*k` f32.
    pub(crate) unsafe fn launch_gemm_f32(
        &self,
        d_weight: &CudaSlice<f32>,
        d_input: &CudaSlice<f32>,
        d_output: &mut CudaSlice<f32>,
        n_rows: u32,
        m: u32,
        k: u32,
    ) -> Result<(), CudaGraphError> {
        // grid = (ceil(N/64), ceil(M/64), 1), block = (16, 16, 1).
        let cfg = LaunchConfig {
            grid_dim: (n_rows.div_ceil(64), m.div_ceil(64), 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };
        self.stream
            .launch_builder(&self.modules.gemm_f32)
            .arg(d_weight)
            .arg(d_input)
            .arg(d_output)
            .arg(&n_rows)
            .arg(&m)
            .arg(&k)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("gemm_f32 launch: {e}")))
    }

    /// Launch `gemm_tq2` on the default stream.
    ///
    /// # Safety
    /// Caller must ensure all slices are valid device pointers on `self.stream`.
    pub(crate) unsafe fn launch_gemm_tq2(
        &self,
        d_weight: &CudaSlice<u8>,
        d_input: &CudaSlice<f32>,
        d_output: &mut CudaSlice<f32>,
        n_rows: u32,
        m: u32,
        k: u32,
    ) -> Result<(), CudaGraphError> {
        // grid = (ceil(N/128), ceil(M/128), 1), block = (16, 16, 1) — each block
        // owns a 128x128 output tile (8x8 micro-tile per thread).
        // Shared mem is static __shared__ in the kernel (8 KiB), so 0 here.
        let cfg = LaunchConfig {
            grid_dim: (n_rows.div_ceil(128), m.div_ceil(128), 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };
        self.stream
            .launch_builder(&self.modules.gemm_tq2)
            .arg(d_weight)
            .arg(d_input)
            .arg(d_output)
            .arg(&n_rows)
            .arg(&m)
            .arg(&k)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("gemm_tq2 launch: {e}")))
    }

    /// Execute a batched **f32-exact** GEMM: `output = input × weightᵀ`.
    ///
    /// CUDA-core sibling of the Metal [`MetalGraph::encode_gemm_f32`] — the
    /// image-generation (FLUX.2) text-encoder f32 path. Dispatches the `gemm_f32`
    /// kernel (plain CUDA-core 4×4 micro-tile, f32 accumulate), numerically
    /// equivalent to the CPU `oxibonsai-image::gemm::gemm_abt` (cos ≈ 1.0).
    ///
    /// # Layout
    ///
    /// `input` / `output` are **column-major** with the batch as the outer
    /// dimension (== row-major `[M,K]` / `[M,N]`): `input[m*k + e]`,
    /// `output[m*n_rows + row]`. `weight` is the pre-uploaded **row-major f32**
    /// `[N,K]` device buffer, so a caller holding row-major `input[M,K]`, f32
    /// `weight[N,K]`, and `out[M,N]` (computing `out = input · weightᵀ`) can pass
    /// its buffers directly — no transpose.
    ///
    /// # Parameters
    /// - `weight`: device `[N, K]` row-major f32 (`n_rows*k` floats).
    /// - `input`: length `m*k` f32.
    /// - `output`: length `m*n_rows` f32 (overwritten).
    /// - `m`: batch size (rows of `A`); arbitrary, including `> 8`.
    /// - `n_rows`: `N`, the number of weight rows.
    /// - `k`: inner dim; arbitrary (`≥ 1`, no `% 128` constraint — the kernel
    ///   zero-clamps its last K-tile).
    ///
    /// # Errors
    /// Returns [`CudaGraphError::DriverError`] for an `input.len() != m*k` /
    /// `output.len() != m*n_rows` mismatch (the CUDA path has no dedicated
    /// dimension-error variant; the Metal sibling uses `InvalidDimensions`), or
    /// for any driver / launch failure.
    pub fn encode_gemm_f32(
        &self,
        weight: &Arc<CudaSlice<f32>>,
        input: &[f32],
        output: &mut [f32],
        m: usize,
        n_rows: usize,
        k: usize,
    ) -> Result<(), CudaGraphError> {
        // ── Validate ─────────────────────────────────────────────────────
        let expected_in = m.checked_mul(k).ok_or_else(|| {
            CudaGraphError::DriverError(format!("encode_gemm_f32: m*k overflow (m={m}, k={k})"))
        })?;
        if input.len() != expected_in {
            return Err(CudaGraphError::DriverError(format!(
                "encode_gemm_f32: input len {} != m*k {expected_in} (m={m}, k={k})",
                input.len()
            )));
        }
        let expected_out = m.checked_mul(n_rows).ok_or_else(|| {
            CudaGraphError::DriverError(format!(
                "encode_gemm_f32: m*n_rows overflow (m={m}, n_rows={n_rows})"
            ))
        })?;
        if output.len() != expected_out {
            return Err(CudaGraphError::DriverError(format!(
                "encode_gemm_f32: output len {} != m*n_rows {expected_out} (m={m}, n_rows={n_rows})",
                output.len()
            )));
        }

        // Degenerate empty GEMM: nothing to do.
        if expected_in == 0 || expected_out == 0 {
            return Ok(());
        }

        // PERF: per-call alloc; add a grow-to-fit pool (cf. TernaryGemvBuffers)
        // in the hardware phase.
        let d_input = self
            .stream
            .clone_htod(input)
            .map_err(|e| CudaGraphError::DriverError(format!("clone_htod gemm_f32 input: {e}")))?;
        let mut d_output = self.stream.alloc_zeros::<f32>(expected_out).map_err(|e| {
            CudaGraphError::DriverError(format!("alloc_zeros gemm_f32 output: {e}"))
        })?;

        unsafe {
            self.launch_gemm_f32(
                weight,
                &d_input,
                &mut d_output,
                n_rows as u32,
                m as u32,
                k as u32,
            )?;
        }

        self.stream.memcpy_dtoh(&d_output, output).map_err(|e| {
            CudaGraphError::DriverError(format!("memcpy_dtoh gemm_f32 output: {e}"))
        })?;
        self.stream
            .synchronize()
            .map_err(|e| CudaGraphError::DriverError(format!("stream sync gemm_f32: {e}")))?;
        Ok(())
    }

    /// Execute a batched **ternary (TQ2_0_g128)** GEMM: `output = input × dequant(weight)ᵀ`.
    ///
    /// CUDA-core sibling of the Metal [`MetalGraph::encode_gemm_tq2`] — the
    /// image-generation (FLUX.2) DiT ternary path. Dispatches the `gemm_tq2`
    /// kernel (plain CUDA-core 2×2 micro-tile, f16-exact `code×scale` decode,
    /// f32 accumulate), numerically equivalent to the CPU dequant + `gemm_abt`.
    ///
    /// # Layout
    ///
    /// `input` / `output` are **column-major** (== row-major `[M,K]` / `[M,N]`):
    /// `input[m*k + e]`, `output[m*n_rows + row]`. `weight` is the pre-uploaded
    /// **SoA** TQ2 device buffer from
    /// [`get_or_upload_weight_tq2_soa`](Self::get_or_upload_weight_tq2_soa)
    /// (`[N·(K/128)·2 B scales][·32 B qs]`).
    ///
    /// # Parameters
    /// - `weight`: pre-uploaded SoA TQ2 device buffer (`N` rows × `k` cols).
    /// - `input`: length `m*k` f32.
    /// - `output`: length `m*n_rows` f32 (overwritten).
    /// - `m`: batch size (rows of `A`); arbitrary, including `> 8`.
    /// - `n_rows`: `N`, the number of weight rows.
    /// - `k`: inner dim; **must** be a multiple of 128.
    ///
    /// # Errors
    /// Returns [`CudaGraphError::DriverError`] if `k % 128 != 0`,
    /// `input.len() != m*k`, `output.len() != m*n_rows`, or for any driver /
    /// launch failure. (The CUDA path has no dedicated dimension-error variant;
    /// the Metal sibling uses `InvalidDimensions`.)
    pub fn encode_gemm_tq2(
        &self,
        weight: &Arc<CudaSlice<u8>>,
        input: &[f32],
        output: &mut [f32],
        m: usize,
        n_rows: usize,
        k: usize,
    ) -> Result<(), CudaGraphError> {
        // ── Validate ─────────────────────────────────────────────────────
        if k % 128 != 0 {
            return Err(CudaGraphError::DriverError(format!(
                "encode_gemm_tq2: k must be a multiple of 128, got {k}"
            )));
        }
        let expected_in = m.checked_mul(k).ok_or_else(|| {
            CudaGraphError::DriverError(format!("encode_gemm_tq2: m*k overflow (m={m}, k={k})"))
        })?;
        if input.len() != expected_in {
            return Err(CudaGraphError::DriverError(format!(
                "encode_gemm_tq2: input len {} != m*k {expected_in} (m={m}, k={k})",
                input.len()
            )));
        }
        let expected_out = m.checked_mul(n_rows).ok_or_else(|| {
            CudaGraphError::DriverError(format!(
                "encode_gemm_tq2: m*n_rows overflow (m={m}, n_rows={n_rows})"
            ))
        })?;
        if output.len() != expected_out {
            return Err(CudaGraphError::DriverError(format!(
                "encode_gemm_tq2: output len {} != m*n_rows {expected_out} (m={m}, n_rows={n_rows})",
                output.len()
            )));
        }

        // Degenerate empty GEMM: nothing to do.
        if expected_in == 0 || expected_out == 0 {
            return Ok(());
        }

        // PERF: per-call alloc; add a grow-to-fit pool (cf. TernaryGemvBuffers)
        // in the hardware phase.
        let d_input = self
            .stream
            .clone_htod(input)
            .map_err(|e| CudaGraphError::DriverError(format!("clone_htod gemm_tq2 input: {e}")))?;
        let mut d_output = self.stream.alloc_zeros::<f32>(expected_out).map_err(|e| {
            CudaGraphError::DriverError(format!("alloc_zeros gemm_tq2 output: {e}"))
        })?;

        unsafe {
            self.launch_gemm_tq2(
                weight,
                &d_input,
                &mut d_output,
                n_rows as u32,
                m as u32,
                k as u32,
            )?;
        }

        self.stream.memcpy_dtoh(&d_output, output).map_err(|e| {
            CudaGraphError::DriverError(format!("memcpy_dtoh gemm_tq2 output: {e}"))
        })?;
        self.stream
            .synchronize()
            .map_err(|e| CudaGraphError::DriverError(format!("stream sync gemm_tq2: {e}")))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::CudaGraph;

    /// Tiny CPU reference: `out[M,N] = A[M,K] · W[N,K]ᵀ`, column-major
    /// (== row-major `[M,N]`), matching the kernel's `outputs[col*n_rows + row]`.
    /// `a` is row-major `[M,K]`, `w` is row-major `[N,K]`.
    fn gemm_abt(a: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut out = vec![0f32; m * n];
        for mm in 0..m {
            for nn in 0..n {
                let mut acc = 0f32;
                for kk in 0..k {
                    acc += a[mm * k + kk] * w[nn * k + kk];
                }
                out[mm * n + nn] = acc;
            }
        }
        out
    }

    /// Cosine similarity of two equal-length vectors (1.0 == identical direction).
    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0f64;
        let mut na = 0f64;
        let mut nb = 0f64;
        for (&x, &y) in a.iter().zip(b.iter()) {
            dot += x as f64 * y as f64;
            na += x as f64 * x as f64;
            nb += y as f64 * y as f64;
        }
        if na == 0.0 || nb == 0.0 {
            return 0.0;
        }
        (dot / (na.sqrt() * nb.sqrt())) as f32
    }

    fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .fold(0f32, |acc, (&x, &y)| acc.max((x - y).abs()))
    }

    /// Parity for the f32-exact `gemm_f32` over a shape sweep that **includes
    /// M > 8** (33, 40) — proving the FULL-M (grid.y + clamp) path computes every
    /// column (no cap-of-8). Skips gracefully if no CUDA device is present.
    #[test]
    fn gemm_f32_parity() {
        let _serial = super::super::types::gpu_parity_test_guard();
        let g = match CudaGraph::global() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("no GPU, skip: {e}");
                return;
            }
        };

        // (M, N, K) — non-tile-multiples + M>8 exercise the clamps & full-M.
        let shapes = [
            (33usize, 40usize, 64usize),
            (1, 1, 128),
            (64, 64, 32),
            (40, 256, 512),
        ];

        for (shape_idx, (m, n, k)) in shapes.into_iter().enumerate() {
            // Distinct upload-cache key per shape (so the cache never collides).
            let weight_key = 7_740_000u64 + shape_idx as u64;
            // Deterministic LCG fill (no rand crate; pattern from
            // metal_graph/tests_gemm_tq2.rs).
            let mut lcg: u32 = 0x1357_9BDF ^ ((m as u32) << 16) ^ ((n as u32) << 8) ^ (k as u32);
            let mut next = || {
                lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((lcg >> 8) & 0xFFFF) as f32 / 65535.0 - 0.5
            };

            let a: Vec<f32> = (0..m * k).map(|_| next()).collect();
            let w: Vec<f32> = (0..n * k).map(|_| next()).collect();

            // Upload the row-major [N, K] f32 weight via the public cache path
            // (distinct key per shape so the cache never collides).
            let d_weight = g
                .get_or_upload_f32_weight(weight_key, &w)
                .expect("get_or_upload_f32_weight failed");

            let mut got = vec![0f32; m * n];
            g.encode_gemm_f32(&d_weight, &a, &mut got, m, n, k)
                .expect("encode_gemm_f32 failed");

            let expected = gemm_abt(&a, &w, m, n, k);

            let cos = cosine(&expected, &got);
            let mae = max_abs_err(&expected, &got);
            assert!(
                cos >= 0.999,
                "gemm_f32 M={m} N={n} K={k}: cos {cos} < 0.999"
            );
            assert!(
                mae < 1e-3,
                "gemm_f32 M={m} N={n} K={k}: max-abs {mae} >= 1e-3"
            );
            assert!(
                got.iter().any(|&v| v.abs() > 1e-6),
                "gemm_f32 M={m} N={n} K={k}: all-zero output (suspicious)"
            );
            eprintln!("gemm_f32: M={m:>3} N={n:>3} K={k:>4} cos={cos:.6} max_abs={mae:e}");
        }
    }

    /// Parity for the ternary `gemm_tq2` over shapes with `K % 128 == 0` and
    /// **M ∈ {40, 96}** (both > 8) — proving the cap-of-8 fix. Builds a random
    /// `BlockTQ2_0_g128` weight (deterministic LCG codes, matching the
    /// metal_graph/tests_gemm_tq2.rs packing), uploads via
    /// `get_or_upload_weight_tq2_soa`, and compares against host-dequant +
    /// `gemm_abt`. Skips gracefully if no CUDA device is present.
    #[test]
    fn gemm_tq2_parity() {
        let _serial = super::super::types::gpu_parity_test_guard();
        use half::f16;
        use oxibonsai_core::BlockTQ2_0_g128;

        let g = match CudaGraph::global() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("no GPU, skip: {e}");
                return;
            }
        };

        // (M, N, K) — K%128==0; M∈{40,96} are both >8 (cap-of-8 proof). Includes
        // a non-tile-multiple N=40 to exercise the boundary clamp.
        let shapes = [(40usize, 40usize, 128usize), (96, 64, 256), (40, 128, 256)];

        for (shape_idx, (m, n_rows, k)) in shapes.into_iter().enumerate() {
            // Distinct upload-cache key per shape (so the cache never collides).
            let weight_key = 7_730_000u64 + shape_idx as u64;
            let blocks_per_row = k / 128;

            // Deterministic LCG codes in {0,1,2} (== {-1,0,+1} after decode),
            // packed LSB-first 4 codes/byte — exactly the metal test packing.
            let mut lcg: u32 = 0x2545_F491 ^ ((n_rows as u32) << 8) ^ (k as u32);
            let mut next_code = || {
                lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((lcg >> 16) % 3) as u8
            };

            let mut blocks: Vec<BlockTQ2_0_g128> = Vec::with_capacity(n_rows * blocks_per_row);
            for row in 0..n_rows {
                for bk in 0..blocks_per_row {
                    let mut qs = [0u8; 32];
                    for b in qs.iter_mut() {
                        let c0 = next_code();
                        let c1 = next_code();
                        let c2 = next_code();
                        let c3 = next_code();
                        *b = c0 | (c1 << 2) | (c2 << 4) | (c3 << 6);
                    }
                    blocks.push(BlockTQ2_0_g128 {
                        qs,
                        d: f16::from_f32(0.05 + 0.003 * (row % 17) as f32 + 0.002 * bk as f32),
                    });
                }
            }

            // Host reference: dequantize the weight to f32 [N, K] once.
            let mut dequant_w = vec![0f32; n_rows * k];
            BlockTQ2_0_g128::dequant(&blocks, &mut dequant_w)
                .expect("dequant reference weight failed");

            // Upload via the public SoA cache path (AoS bytes → SoA inside).
            let aos_bytes = {
                let ptr = blocks.as_ptr() as *const u8;
                let len = std::mem::size_of_val(blocks.as_slice());
                unsafe { std::slice::from_raw_parts(ptr, len) }
            };
            let d_weight = g
                .get_or_upload_weight_tq2_soa(weight_key, aos_bytes)
                .expect("get_or_upload_weight_tq2_soa failed");

            // Deterministic, index-derived input [M, K] (row-major).
            let input: Vec<f32> = (0..m * k)
                .map(|i| {
                    let row = i / k;
                    let col = i % k;
                    ((col as f32) * 0.011 - 0.37).sin() + (row as f32) * 0.0005
                })
                .collect();

            let mut got = vec![0f32; m * n_rows];
            g.encode_gemm_tq2(&d_weight, &input, &mut got, m, n_rows, k)
                .expect("encode_gemm_tq2 failed");

            let expected = gemm_abt(&input, &dequant_w, m, n_rows, k);

            let cos = cosine(&expected, &got);
            let mae = max_abs_err(&expected, &got);
            assert!(
                cos >= 0.999,
                "gemm_tq2 M={m} N={n_rows} K={k}: cos {cos} < 0.999"
            );
            assert!(
                mae < 1e-3,
                "gemm_tq2 M={m} N={n_rows} K={k}: max-abs {mae} >= 1e-3"
            );
            assert!(
                got.iter().any(|&v| v.abs() > 1e-6),
                "gemm_tq2 M={m} N={n_rows} K={k}: all-zero output (suspicious)"
            );
            eprintln!("gemm_tq2: M={m:>3} N={n_rows:>3} K={k:>4} cos={cos:.6} max_abs={mae:e}");
        }
    }
}
