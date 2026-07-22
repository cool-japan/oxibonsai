//! # CudaGraph — FLUX.2 VAE decoder per-op f32 primitives
//!
//! Public `CudaGraph::encode_*` entry points for the FLUX.2 VAE-decoder f32 GPU
//! primitives — the CUDA mirror of `metal_graph/vae.rs`. Each method uploads its
//! host buffers, launches the matching kernel from
//! [`super::super::cuda_imagen_vae_kernels`] (`CUDA_IMAGEN_VAE_SRC`), then
//! downloads and synchronizes. Parity is validated against self-contained CPU
//! ports of `oxibonsai-image::vae` (`conv.rs`, `norm.rs`, `ops.rs`):
//!
//! - [`CudaGraph::encode_conv2d_f32`] — stride-1 "same"-padded Conv2d. For
//!   `k=1, pad=0` it is a pure channel-mix that reuses the parity-clean
//!   [`encode_gemm_f32`](CudaGraph::encode_gemm_f32). For `k≥3` it builds the
//!   im2col patch matrix on-device (`im2col_f32`, `(kH,kW,C_in)` order matching
//!   the MLX weight `[C_out,kH,kW,C_in]` flattening), tiled over output rows, and
//!   feeds each tile to `encode_gemm_f32`.
//! - [`CudaGraph::encode_groupnorm_f32`] — GroupNorm (32 groups, eps 1e-6, f64
//!   reduction).
//! - [`CudaGraph::encode_silu_f32`] — element-wise SiLU.
//! - [`CudaGraph::encode_upsample_nearest_f32`] — nearest ×2 upsample.
//!
//! All buffers are flat NCHW `[C, H, W]` f32 (batch 1).
//!
//! NOTE (Linux integration): this prototype is authored blind on macOS and is
//! gated off there. It cannot be compiled until the CUDA toolkit is available.

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use std::sync::Arc;

use super::types::CudaGraphError;

use super::cudagraph_type::CudaGraph;

/// Cap (bytes) on the per-tile im2col patch buffer for the k≥3 conv path.
///
/// Mirrors `metal_graph/vae.rs::IM2COL_TILE_CAP_BYTES`. The full im2col matrix
/// for the largest VAE conv (up2, `[262144, 3456]`) is ~3.6 GB, far too big to
/// materialize at once. Output rows are tiled so the patch buffer never exceeds
/// this cap (~256 MiB).
const IM2COL_TILE_CAP_BYTES: usize = 256 * 1024 * 1024;

impl CudaGraph {
    /// Stride-1, "same"-padded 2-D convolution on an NCHW input `[C_in, H, W]`
    /// (batch 1), returning the NCHW output `[C_out, H_out, W_out]` (`H_out = H +
    /// 2·pad − k + 1`), with the MLX-layout weight `[C_out, kH, kW, C_in]`
    /// (flattened row-major == `[C_out, kH·kW·C_in]`) and a per-output-channel
    /// `bias[C_out]`.
    ///
    /// # Algorithm (mirrors `metal_graph/vae.rs::encode_conv2d_f32`)
    ///
    /// - **`k == 1 && pad == 0`** — pure channel-mix: reshape the input
    ///   `[C_in, H·W]` → `[H·W, C_in]`, run the parity-clean
    ///   [`encode_gemm_f32`](CudaGraph::encode_gemm_f32) (`m = H·W`, `n = C_out`,
    ///   `k = C_in`), then transpose `[H·W, C_out]` → `[C_out, H·W]` and add bias.
    /// - **otherwise (`k ≥ 1`, `pad ≥ 0`)** — build the im2col patch matrix
    ///   `[rows, kH·kW·C_in]` on-device in the `(kH,kW,C_in)` order that matches
    ///   the weight flattening (`im2col_f32`), tiled over output rows so the
    ///   patch buffer stays ≤ `IM2COL_TILE_CAP_BYTES`; each tile's patches are
    ///   downloaded and fed to [`encode_gemm_f32`](CudaGraph::encode_gemm_f32),
    ///   then the `[rows, C_out]` result is scattered into the NCHW output with
    ///   the bias.
    ///
    /// The element order is identical to `oxibonsai-image::vae::conv` (im2col +
    /// GEMM), so the result is bit-compatible up to f32 sum reassociation.
    ///
    /// # Parameters
    /// - `weight`: pre-uploaded row-major f32 weight handle, `[C_out, kH·kW·C_in]`
    ///   (e.g. via [`get_or_upload_f32_weight`](CudaGraph::get_or_upload_f32_weight)).
    /// - `input`: NCHW `[C_in, H, W]`, length `c_in * h * w`.
    /// - `bias`: `[C_out]`.
    /// - `output`: NCHW `[C_out, H_out, W_out]` (overwritten), length
    ///   `c_out * h_out * w_out`.
    /// - `c_in` / `c_out`: input / output channels.
    /// - `h` / `w`: input spatial dims.
    /// - `k`: kernel edge (square).
    /// - `pad`: spatial padding (same on all sides).
    ///
    /// # Errors
    /// [`CudaGraphError::DriverError`] on any length / shape mismatch (or
    /// degenerate `h_out`/`w_out`), or a buffer / launch error.
    ///
    /// PERF: per-call alloc / per-tile D2H of patches + reuse of `encode_gemm_f32`
    /// (correctness over fused-stream perf); pool buffers + fuse im2col→gemm on a
    /// single stream in the hardware phase. DEFER the implicit-GEMM fused conv.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_conv2d_f32(
        &self,
        weight: &Arc<CudaSlice<f32>>,
        input: &[f32],
        bias: &[f32],
        output: &mut [f32],
        c_in: usize,
        c_out: usize,
        h: usize,
        w: usize,
        k: usize,
        pad: usize,
    ) -> Result<(), CudaGraphError> {
        // ── Validate ─────────────────────────────────────────────────────
        if k == 0 {
            return Err(CudaGraphError::DriverError(
                "encode_conv2d_f32: kernel size k must be >= 1".into(),
            ));
        }
        let expected_in = c_in
            .checked_mul(h)
            .and_then(|x| x.checked_mul(w))
            .ok_or_else(|| {
                CudaGraphError::DriverError(format!(
                    "encode_conv2d_f32: c_in*h*w overflow (c_in={c_in}, h={h}, w={w})"
                ))
            })?;
        if input.len() != expected_in {
            return Err(CudaGraphError::DriverError(format!(
                "encode_conv2d_f32: input len {} != c_in*h*w {expected_in}",
                input.len()
            )));
        }
        if bias.len() != c_out {
            return Err(CudaGraphError::DriverError(format!(
                "encode_conv2d_f32: bias len {} != c_out {c_out}",
                bias.len()
            )));
        }
        // "same"-stride-1 output geometry: h_out = h + 2*pad - k + 1.
        let h_pad = h + 2 * pad;
        let w_pad = w + 2 * pad;
        if h_pad + 1 < k || w_pad + 1 < k {
            return Err(CudaGraphError::DriverError(format!(
                "encode_conv2d_f32: kernel {k} larger than padded input {h_pad}x{w_pad}"
            )));
        }
        let h_out = h_pad + 1 - k;
        let w_out = w_pad + 1 - k;
        let spatial = h_out.checked_mul(w_out).ok_or_else(|| {
            CudaGraphError::DriverError(format!(
                "encode_conv2d_f32: h_out*w_out overflow (h_out={h_out}, w_out={w_out})"
            ))
        })?;
        let expected_out = c_out.checked_mul(spatial).ok_or_else(|| {
            CudaGraphError::DriverError(format!(
                "encode_conv2d_f32: c_out*spatial overflow (c_out={c_out}, spatial={spatial})"
            ))
        })?;
        if output.len() != expected_out {
            return Err(CudaGraphError::DriverError(format!(
                "encode_conv2d_f32: output len {} != c_out*h_out*w_out {expected_out}",
                output.len()
            )));
        }
        if expected_in == 0 || expected_out == 0 {
            return Ok(());
        }

        // ── Fast path: k=1, pad=0 → pure channel-mix via encode_gemm_f32. ──
        if k == 1 && pad == 0 {
            // Reshape NCHW [C_in, H·W] → spatial-major [H·W, C_in].
            let hw = h * w; // == spatial (h_out=h, w_out=w when k=1,pad=0)
            let mut reshaped = vec![0f32; hw * c_in];
            for ci in 0..c_in {
                let src = &input[ci * hw..(ci + 1) * hw];
                for (s, &v) in src.iter().enumerate() {
                    reshaped[s * c_in + ci] = v;
                }
            }
            // GEMM: out_spatial[hw, c_out] = reshaped[hw, c_in] · weightᵀ.
            let mut out_spatial = vec![0f32; hw * c_out];
            self.encode_gemm_f32(weight, &reshaped, &mut out_spatial, hw, c_out, c_in)?;
            // Transpose [hw, c_out] → NCHW [c_out, hw] and add bias.
            for oc in 0..c_out {
                let b = bias[oc];
                let dst = &mut output[oc * hw..(oc + 1) * hw];
                for (s, slot) in dst.iter_mut().enumerate() {
                    *slot = out_spatial[s * c_out + oc] + b;
                }
            }
            return Ok(());
        }

        // ── k ≥ 1, pad ≥ 0: contraction dim + weight-size validation. ──
        let patch_dim = k
            .checked_mul(k)
            .and_then(|x| x.checked_mul(c_in))
            .ok_or_else(|| {
                CudaGraphError::DriverError(format!(
                    "encode_conv2d_f32: patch_dim overflow (k={k}, c_in={c_in})"
                ))
            })?;
        // Verify the weight buffer is large enough for [c_out, patch_dim] f32.
        let weight_floats = c_out.checked_mul(patch_dim).ok_or_else(|| {
            CudaGraphError::DriverError(format!(
                "encode_conv2d_f32: c_out*patch_dim overflow (c_out={c_out}, patch_dim={patch_dim})"
            ))
        })?;
        if weight.len() < weight_floats {
            return Err(CudaGraphError::DriverError(format!(
                "encode_conv2d_f32: weight handle holds {} f32 < c_out*patch_dim {weight_floats}",
                weight.len()
            )));
        }

        // ── General path: GPU im2col (tiled) → encode_gemm_f32 per tile. ──
        self.encode_conv2d_f32_im2col(
            weight, input, bias, output, c_in, c_out, h, w, k, pad, spatial, patch_dim, w_out,
        )
    }

    /// Tiled-im2col conv2d: launch `im2col_f32` to materialize a capped
    /// `[tile_rows, patch_dim]` patch buffer per tile (device-resident), then run
    /// the parity-clean `gemm_f32` kernel straight off those device patches
    /// ([`launch_gemm_f32`](Self::launch_gemm_f32)); scatter each tile's
    /// `[rows, C_out]` result into the NCHW output with the bias. Output rows are
    /// tiled so the patch buffer stays ≤ `IM2COL_TILE_CAP_BYTES`. Mirrors
    /// `metal_graph/vae.rs::encode_conv2d_f32_im2col`.
    ///
    /// Preconditions (validated by the caller [`encode_conv2d_f32`]): `spatial =
    /// h_out·w_out`, `patch_dim = k·k·c_in`, the weight handle holds ≥
    /// `c_out·patch_dim` f32, and `input`/`output` lengths are `c_in·h·w` /
    /// `c_out·spatial`.
    ///
    /// The im2col patches stay RESIDENT on the device — `im2col_f32` and the GEMM
    /// run back-to-back on `self.stream`, so the (up to `IM2COL_TILE_CAP_BYTES`)
    /// patch matrix is never round-tripped to the host; only the much smaller
    /// per-tile `[rows, C_out]` result is downloaded. (PERF still open: pool the
    /// input/patch/output device buffers across calls.)
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_conv2d_f32_im2col(
        &self,
        weight: &Arc<CudaSlice<f32>>,
        input: &[f32],
        bias: &[f32],
        output: &mut [f32],
        c_in: usize,
        c_out: usize,
        h: usize,
        w: usize,
        k: usize,
        pad: usize,
        spatial: usize,
        patch_dim: usize,
        w_out: usize,
    ) -> Result<(), CudaGraphError> {
        // Upload the NCHW input once (hoisted out of the tile loop).
        let d_input = self
            .stream
            .clone_htod(input)
            .map_err(|e| CudaGraphError::DriverError(format!("clone_htod conv input: {e}")))?;

        // Tile the output rows so the patch buffer stays ≤ the cap.
        let patch_row_bytes = patch_dim
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                CudaGraphError::DriverError(
                    "encode_conv2d_f32: patch row byte size overflow".into(),
                )
            })?;
        let tile_rows = (IM2COL_TILE_CAP_BYTES / patch_row_bytes.max(1))
            .max(1)
            .min(spatial);

        // Device patch scratch [tile_rows, patch_dim]; reused across tiles.
        let patches_floats = tile_rows.checked_mul(patch_dim).ok_or_else(|| {
            CudaGraphError::DriverError("encode_conv2d_f32: tile_rows*patch_dim overflow".into())
        })?;
        let mut d_patches = self
            .stream
            .alloc_zeros::<f32>(patches_floats)
            .map_err(|e| CudaGraphError::DriverError(format!("alloc_zeros patches: {e}")))?;

        // Device GEMM-output scratch [tile_rows, c_out]; reused across tiles. The
        // patches stay RESIDENT: the GEMM reads `d_patches` directly, so the
        // (huge) patch matrix is never round-tripped to the host — only the much
        // smaller (c_out << patch_dim) per-tile result is downloaded.
        let out_tile_floats = tile_rows.checked_mul(c_out).ok_or_else(|| {
            CudaGraphError::DriverError("encode_conv2d_f32: tile_rows*c_out overflow".into())
        })?;
        let mut d_out_tile = self
            .stream
            .alloc_zeros::<f32>(out_tile_floats)
            .map_err(|e| CudaGraphError::DriverError(format!("alloc_zeros conv out tile: {e}")))?;

        // Host staging for the (small) downloaded GEMM output tile only.
        let mut out_tile = vec![0f32; tile_rows * c_out];

        let mut row_start = 0usize;
        while row_start < spatial {
            let rows = (spatial - row_start).min(tile_rows);
            let n_elems = rows.checked_mul(patch_dim).ok_or_else(|| {
                CudaGraphError::DriverError("encode_conv2d_f32: rows*patch_dim overflow".into())
            })? as u32;

            // im2col → d_patches[rows, patch_dim], then the f32 GEMM straight off
            // those resident patches → d_out_tile[rows, c_out] = patches · weightᵀ.
            // Both launch on `self.stream`, which serializes them, so no
            // intermediate sync (and, crucially, no patch D2H/H2D) is needed.
            unsafe {
                self.launch_imagen_vae_im2col(
                    &d_input,
                    &mut d_patches,
                    c_in as u32,
                    h as u32,
                    w as u32,
                    k as u32,
                    pad as u32,
                    w_out as u32,
                    row_start as u32,
                    n_elems,
                )?;
                // SAFETY: `d_patches` holds tile_rows*patch_dim ≥ rows*patch_dim
                // f32 (just written by im2col above for the [0, rows*patch_dim)
                // range the GEMM reads), `d_out_tile` holds tile_rows*c_out ≥
                // rows*c_out f32, and `weight` holds ≥ c_out*patch_dim f32
                // (validated by the caller). All live on `self.stream`, on which
                // the GEMM is ordered after the im2col.
                self.launch_gemm_f32(
                    weight,
                    &d_patches,
                    &mut d_out_tile,
                    c_out as u32,
                    rows as u32,
                    patch_dim as u32,
                )?;
            }

            // Download ONLY the [rows, c_out] result tile (row-major
            // out_tile[m*c_out + oc]).
            let used_out = rows * c_out;
            {
                let d_view = d_out_tile.slice(0..used_out);
                self.stream
                    .memcpy_dtoh(&d_view, &mut out_tile[..used_out])
                    .map_err(|e| {
                        CudaGraphError::DriverError(format!("memcpy_dtoh conv out tile: {e}"))
                    })?;
            }
            self.stream
                .synchronize()
                .map_err(|e| CudaGraphError::DriverError(format!("sync conv out D2H: {e}")))?;

            // Scatter into NCHW output with bias.
            // out_tile is row-major [rows, c_out] (out_tile[m*c_out + oc]).
            for local_row in 0..rows {
                let out_idx = row_start + local_row;
                let base = local_row * c_out;
                for oc in 0..c_out {
                    output[oc * spatial + out_idx] = out_tile[base + oc] + bias[oc];
                }
            }

            row_start += rows;
        }

        Ok(())
    }

    /// PyTorch-compatible GroupNorm on an NCHW buffer `[C, H, W]` (batch 1), in
    /// place: split `C` into `num_groups` contiguous groups, normalize each over
    /// all its channels × spatial positions (`(x − mean) / sqrt(var + eps)`,
    /// population variance), then apply the per-channel affine `weight[c]` /
    /// `bias[c]`. Mirrors `oxibonsai-image::vae::norm::forward_inplace` and
    /// `metal_graph/vae.rs::encode_groupnorm_f32`.
    ///
    /// One block per group performs an f64 reduction (CUDA has `double`, so this
    /// matches the f64 CPU reference exactly — tighter than the Metal Kahan-f32).
    ///
    /// # Parameters
    /// - `x`: NCHW `[C, H·W]` (read-write / in-place), length `channels * hw`.
    /// - `weight` / `bias`: per-channel affine `[C]`.
    /// - `channels`: `C` (must be divisible by `num_groups`).
    /// - `hw`: `H·W`.
    /// - `num_groups`: number of groups (32 in the VAE).
    /// - `eps`: epsilon inside the sqrt (1e-6 in the VAE).
    ///
    /// # Errors
    /// [`CudaGraphError::DriverError`] on a length / divisibility mismatch, or a
    /// buffer / launch error.
    ///
    /// PERF: per-call alloc; pool in hardware phase.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_groupnorm_f32(
        &self,
        x: &mut [f32],
        weight: &[f32],
        bias: &[f32],
        channels: usize,
        hw: usize,
        num_groups: usize,
        eps: f32,
    ) -> Result<(), CudaGraphError> {
        if num_groups == 0 || channels % num_groups != 0 {
            return Err(CudaGraphError::DriverError(format!(
                "encode_groupnorm_f32: channels {channels} not divisible by num_groups {num_groups}"
            )));
        }
        let expected = channels.checked_mul(hw).ok_or_else(|| {
            CudaGraphError::DriverError(format!(
                "encode_groupnorm_f32: channels*hw overflow (channels={channels}, hw={hw})"
            ))
        })?;
        if x.len() != expected {
            return Err(CudaGraphError::DriverError(format!(
                "encode_groupnorm_f32: x len {} != channels*hw {expected}",
                x.len()
            )));
        }
        if weight.len() != channels || bias.len() != channels {
            return Err(CudaGraphError::DriverError(format!(
                "encode_groupnorm_f32: weight/bias len ({}/{}) != channels {channels}",
                weight.len(),
                bias.len()
            )));
        }
        if expected == 0 {
            return Ok(());
        }

        let mut d_x = self
            .stream
            .clone_htod(x)
            .map_err(|e| CudaGraphError::DriverError(format!("clone_htod groupnorm x: {e}")))?;
        let d_weight = self
            .stream
            .clone_htod(weight)
            .map_err(|e| CudaGraphError::DriverError(format!("clone_htod groupnorm w: {e}")))?;
        let d_bias = self
            .stream
            .clone_htod(bias)
            .map_err(|e| CudaGraphError::DriverError(format!("clone_htod groupnorm b: {e}")))?;

        unsafe {
            self.launch_imagen_vae_groupnorm(
                &mut d_x,
                &d_weight,
                &d_bias,
                channels as u32,
                hw as u32,
                num_groups as u32,
                eps,
            )?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaGraphError::DriverError(format!("sync groupnorm: {e}")))?;
        self.stream
            .memcpy_dtoh(&d_x, x)
            .map_err(|e| CudaGraphError::DriverError(format!("memcpy_dtoh groupnorm x: {e}")))?;
        self.stream
            .synchronize()
            .map_err(|e| CudaGraphError::DriverError(format!("sync groupnorm D2H: {e}")))?;
        Ok(())
    }

    /// Element-wise SiLU (`x · sigmoid(x) = x / (1 + exp(-x))`), in place over a
    /// flat f32 buffer. Mirrors `oxibonsai-image::math::silu` and
    /// `metal_graph/vae.rs::encode_silu_f32`.
    ///
    /// # Errors
    /// [`CudaGraphError::DriverError`] if the GPU work cannot be launched.
    ///
    /// PERF: per-call alloc; pool in hardware phase.
    pub fn encode_silu_f32(&self, x: &mut [f32]) -> Result<(), CudaGraphError> {
        if x.is_empty() {
            return Ok(());
        }
        let len = x.len();
        let mut d_x = self
            .stream
            .clone_htod(x)
            .map_err(|e| CudaGraphError::DriverError(format!("clone_htod silu x: {e}")))?;

        unsafe {
            self.launch_imagen_vae_silu(&mut d_x, len as u32)?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaGraphError::DriverError(format!("sync silu: {e}")))?;
        self.stream
            .memcpy_dtoh(&d_x, x)
            .map_err(|e| CudaGraphError::DriverError(format!("memcpy_dtoh silu x: {e}")))?;
        self.stream
            .synchronize()
            .map_err(|e| CudaGraphError::DriverError(format!("sync silu D2H: {e}")))?;
        Ok(())
    }

    /// Nearest-neighbour ×2 upsample of an NCHW buffer `[C, H, W]` → `[C, 2H,
    /// 2W]` (each pixel repeated 2× along H and W). Mirrors
    /// `oxibonsai-image::vae::ops::upsample_nearest2x` and
    /// `metal_graph/vae.rs::encode_upsample_nearest_f32`.
    ///
    /// # Parameters
    /// - `input`: NCHW `[C, H·W]`, length `c * h * w`.
    /// - `output`: NCHW `[C, 2H·2W]` (overwritten), length `c * 4 * h * w`.
    /// - `c` / `h` / `w`: input channel / spatial dims.
    ///
    /// # Errors
    /// [`CudaGraphError::DriverError`] on a length mismatch, or a buffer / launch
    /// error.
    ///
    /// PERF: per-call alloc; pool in hardware phase.
    pub fn encode_upsample_nearest_f32(
        &self,
        input: &[f32],
        output: &mut [f32],
        c: usize,
        h: usize,
        w: usize,
    ) -> Result<(), CudaGraphError> {
        let expected_in = c
            .checked_mul(h)
            .and_then(|x| x.checked_mul(w))
            .ok_or_else(|| {
                CudaGraphError::DriverError(format!(
                    "encode_upsample_nearest_f32: c*h*w overflow (c={c}, h={h}, w={w})"
                ))
            })?;
        if input.len() != expected_in {
            return Err(CudaGraphError::DriverError(format!(
                "encode_upsample_nearest_f32: input len {} != c*h*w {expected_in}",
                input.len()
            )));
        }
        let expected_out = expected_in.checked_mul(4).ok_or_else(|| {
            CudaGraphError::DriverError("encode_upsample_nearest_f32: output size overflow".into())
        })?;
        if output.len() != expected_out {
            return Err(CudaGraphError::DriverError(format!(
                "encode_upsample_nearest_f32: output len {} != c*4*h*w {expected_out}",
                output.len()
            )));
        }
        if expected_in == 0 {
            return Ok(());
        }

        let d_input = self
            .stream
            .clone_htod(input)
            .map_err(|e| CudaGraphError::DriverError(format!("clone_htod upsample in: {e}")))?;
        let mut d_output = self
            .stream
            .alloc_zeros::<f32>(expected_out)
            .map_err(|e| CudaGraphError::DriverError(format!("alloc_zeros upsample out: {e}")))?;

        unsafe {
            self.launch_imagen_vae_upsample_nearest(
                &d_input,
                &mut d_output,
                c as u32,
                h as u32,
                w as u32,
                expected_out as u32,
            )?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaGraphError::DriverError(format!("sync upsample: {e}")))?;
        self.stream
            .memcpy_dtoh(&d_output, output)
            .map_err(|e| CudaGraphError::DriverError(format!("memcpy_dtoh upsample out: {e}")))?;
        self.stream
            .synchronize()
            .map_err(|e| CudaGraphError::DriverError(format!("sync upsample D2H: {e}")))?;
        Ok(())
    }

    // ── Raw kernel launchers (one block of 256 threads, default stream) ──────

    /// Launch `im2col_f32` (grid = ⌈n_elems/256⌉, block = 256) on the default
    /// stream.
    ///
    /// # Safety
    /// All slices must be valid device pointers on `self.stream`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_imagen_vae_im2col(
        &self,
        d_input: &CudaSlice<f32>,
        d_patches: &mut CudaSlice<f32>,
        c_in: u32,
        h: u32,
        w: u32,
        k: u32,
        pad: u32,
        w_out: u32,
        row_start: u32,
        n_elems: u32,
    ) -> Result<(), CudaGraphError> {
        let grid_x = n_elems.div_ceil(256);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        self.stream
            .launch_builder(&self.modules.imagen_vae_im2col)
            .arg(d_input)
            .arg(d_patches)
            .arg(&c_in)
            .arg(&h)
            .arg(&w)
            .arg(&k)
            .arg(&pad)
            .arg(&w_out)
            .arg(&row_start)
            .arg(&n_elems)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("im2col_f32 launch: {e}")))
    }

    /// Launch `groupnorm_f32` (grid = num_groups, block = 256) on the default
    /// stream.
    ///
    /// # Safety
    /// All slices must be valid device pointers on `self.stream`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_imagen_vae_groupnorm(
        &self,
        d_x: &mut CudaSlice<f32>,
        d_weight: &CudaSlice<f32>,
        d_bias: &CudaSlice<f32>,
        channels: u32,
        hw: u32,
        num_groups: u32,
        eps: f32,
    ) -> Result<(), CudaGraphError> {
        let cfg = LaunchConfig {
            grid_dim: (num_groups, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        self.stream
            .launch_builder(&self.modules.imagen_vae_groupnorm)
            .arg(d_x)
            .arg(d_weight)
            .arg(d_bias)
            .arg(&channels)
            .arg(&hw)
            .arg(&num_groups)
            .arg(&eps)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("groupnorm_f32 launch: {e}")))
    }

    /// Launch `silu_f32` (grid = ⌈n/256⌉, block = 256) on the default stream.
    ///
    /// # Safety
    /// `d_x` must be a valid device pointer on `self.stream`.
    unsafe fn launch_imagen_vae_silu(
        &self,
        d_x: &mut CudaSlice<f32>,
        n: u32,
    ) -> Result<(), CudaGraphError> {
        let grid_x = n.div_ceil(256);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        self.stream
            .launch_builder(&self.modules.imagen_vae_silu)
            .arg(d_x)
            .arg(&n)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("silu_f32 launch: {e}")))
    }

    /// Launch `upsample_nearest_f32` (grid = ⌈n_out/256⌉, block = 256) on the
    /// default stream.
    ///
    /// # Safety
    /// All slices must be valid device pointers on `self.stream`.
    unsafe fn launch_imagen_vae_upsample_nearest(
        &self,
        d_input: &CudaSlice<f32>,
        d_output: &mut CudaSlice<f32>,
        c: u32,
        h: u32,
        w: u32,
        n_out: u32,
    ) -> Result<(), CudaGraphError> {
        let grid_x = n_out.div_ceil(256);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // The kernel recomputes n_out = c*2h*2w internally; the host bound is
        // only used to size the launch grid.
        self.stream
            .launch_builder(&self.modules.imagen_vae_upsample_nearest)
            .arg(d_input)
            .arg(d_output)
            .arg(&c)
            .arg(&h)
            .arg(&w)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("upsample_nearest_f32 launch: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic bounded f32 fill (`tanh`-ish), seeded by `seed`.
    /// Copied from `metal_graph/tests_vae.rs::vae_fill`.
    fn vae_fill(n: usize, seed: u32) -> Vec<f32> {
        let mut v = vec![0f32; n];
        let mut lcg: u32 = 0x9E37_79B9 ^ seed.wrapping_mul(2_654_435_761);
        for slot in v.iter_mut() {
            lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *slot = ((lcg >> 8) as f32 / (1u32 << 24) as f32) - 0.5;
        }
        v
    }

    /// CPU im2col, mirroring `oxibonsai-image::vae::conv::build_im2col` exactly:
    /// `patches[(oh*w_out+ow)*patch_dim + (kh*k+kw)*c_in + ci]` in `(kh,kw,ci)`
    /// order, zero-padded.
    #[allow(clippy::too_many_arguments)]
    fn cpu_build_im2col(
        input: &[f32],
        patches: &mut [f32],
        in_ch: usize,
        h: usize,
        w: usize,
        k: usize,
        pad: usize,
        h_out: usize,
        w_out: usize,
    ) {
        let patch_dim = k * k * in_ch;
        let hw_plane = h * w;
        for oh in 0..h_out {
            for ow in 0..w_out {
                let row =
                    &mut patches[(oh * w_out + ow) * patch_dim..(oh * w_out + ow + 1) * patch_dim];
                for kh in 0..k {
                    let ih = oh + kh;
                    if ih < pad || ih >= h + pad {
                        continue;
                    }
                    let ih = ih - pad;
                    for kw in 0..k {
                        let iw = ow + kw;
                        if iw < pad || iw >= w + pad {
                            continue;
                        }
                        let iw = iw - pad;
                        let dst_base = (kh * k + kw) * in_ch;
                        let src_base = ih * w + iw;
                        for ci in 0..in_ch {
                            row[dst_base + ci] = input[ci * hw_plane + src_base];
                        }
                    }
                }
            }
        }
    }

    /// Full CPU Conv2d reference (im2col + naive GEMM + transpose + bias),
    /// mirroring `oxibonsai-image::vae::conv::Conv2d::forward`. Returns NCHW
    /// `[c_out, h_out, w_out]`.
    #[allow(clippy::too_many_arguments)]
    fn cpu_conv2d(
        input: &[f32],
        weight: &[f32], // [c_out, k*k*c_in] row-major (MLX [c_out,kH,kW,c_in] flattened)
        bias: &[f32],   // [c_out]
        c_in: usize,
        c_out: usize,
        h: usize,
        w: usize,
        k: usize,
        pad: usize,
    ) -> (Vec<f32>, usize, usize) {
        let h_out = h + 2 * pad + 1 - k;
        let w_out = w + 2 * pad + 1 - k;
        let patch_dim = k * k * c_in;
        let spatial = h_out * w_out;
        let mut patches = vec![0f32; spatial * patch_dim];
        cpu_build_im2col(input, &mut patches, c_in, h, w, k, pad, h_out, w_out);
        let mut out = vec![0f32; c_out * spatial];
        for oc in 0..c_out {
            let w_row = &weight[oc * patch_dim..(oc + 1) * patch_dim];
            let b = bias[oc];
            for hw in 0..spatial {
                let p_row = &patches[hw * patch_dim..(hw + 1) * patch_dim];
                let mut acc = 0f32;
                for kk in 0..patch_dim {
                    acc += p_row[kk] * w_row[kk];
                }
                out[oc * spatial + hw] = acc + b;
            }
        }
        (out, h_out, w_out)
    }

    /// CPU GroupNorm reference, mirroring
    /// `oxibonsai-image::vae::norm::GroupNorm::forward_inplace` (f64 accumulation).
    fn cpu_groupnorm(
        x: &mut [f32],
        weight: &[f32],
        bias: &[f32],
        channels: usize,
        hw: usize,
        num_groups: usize,
        eps: f32,
    ) {
        let gs = channels / num_groups;
        let group_elems = gs * hw;
        let inv_n = 1.0f64 / group_elems as f64;
        for g in 0..num_groups {
            let c0 = g * gs;
            let base = c0 * hw;
            let group = &mut x[base..base + group_elems];
            let mut mean = 0.0f64;
            for &v in group.iter() {
                mean += v as f64;
            }
            mean *= inv_n;
            let mut var = 0.0f64;
            for &v in group.iter() {
                let d = v as f64 - mean;
                var += d * d;
            }
            var *= inv_n;
            let inv_std = (1.0 / (var + eps as f64).sqrt()) as f32;
            let mean_f = mean as f32;
            for ci in 0..gs {
                let c = c0 + ci;
                let wgt = weight[c];
                let bia = bias[c];
                let chan = &mut group[ci * hw..(ci + 1) * hw];
                for v in chan.iter_mut() {
                    *v = (*v - mean_f) * inv_std * wgt + bia;
                }
            }
        }
    }

    /// CPU nearest ×2 upsample, mirroring
    /// `oxibonsai-image::vae::ops::upsample_nearest2x`.
    fn cpu_upsample_nearest(input: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
        let w_out = w * 2;
        let h_out = h * 2;
        let mut out = vec![0f32; c * h_out * w_out];
        for ch in 0..c {
            for ho in 0..h_out {
                for wo in 0..w_out {
                    out[ch * h_out * w_out + ho * w_out + wo] =
                        input[ch * h * w + (ho / 2) * w + (wo / 2)];
                }
            }
        }
        out
    }

    /// CPU SiLU, mirroring `oxibonsai-image::math::silu`.
    fn cpu_silu(x: &[f32]) -> Vec<f32> {
        x.iter().map(|&v| v / (1.0 + (-v).exp())).collect()
    }

    /// Max-abs and relative-L2 error between two equal-length slices.
    fn err_stats(a: &[f32], b: &[f32]) -> (f32, f32) {
        let mut max_abs = 0f32;
        let mut num = 0f64;
        let mut den = 0f64;
        for (&x, &y) in a.iter().zip(b.iter()) {
            let e = (x - y).abs();
            if e > max_abs {
                max_abs = e;
            }
            num += (e as f64) * (e as f64);
            den += (x as f64) * (x as f64);
        }
        let rel_l2 = if den > 0.0 {
            (num.sqrt() / den.sqrt()) as f32
        } else {
            num.sqrt() as f32
        };
        (max_abs, rel_l2)
    }

    /// Acquire the global `CudaGraph`, or return `None` to gracefully skip when
    /// no CUDA device / driver is present (CI on machines without a GPU).
    fn graph_or_skip(what: &str) -> Option<std::sync::Arc<CudaGraph>> {
        match CudaGraph::global() {
            Ok(g) => Some(g),
            Err(_) => {
                eprintln!("no CUDA device — skipping {what}");
                None
            }
        }
    }

    /// Parity: device `im2col_f32` (via a public test driver) vs CPU im2col —
    /// EXACT equality (the gather is an integer permutation, no float math).
    ///
    /// Driven through `encode_conv2d_f32_im2col` indirectly is awkward, so this
    /// drives the raw launcher: build patches on device, download, compare.
    #[test]
    fn im2col_f32_matches_cpu_exact() {
        let _serial = super::super::types::gpu_parity_test_guard();
        let Some(graph) = graph_or_skip("im2col exact parity") else {
            return;
        };
        // (c_in, h, w) with k=3, pad=1.
        let cases = [(3usize, 5usize, 5usize), (3, 8, 8), (8, 5, 5), (8, 8, 8)];
        let k = 3usize;
        let pad = 1usize;
        for &(c_in, h, w) in &cases {
            let h_out = h + 2 * pad + 1 - k;
            let w_out = w + 2 * pad + 1 - k;
            let spatial = h_out * w_out;
            let patch_dim = k * k * c_in;
            let input = vae_fill(c_in * h * w, c_in as u32 * 13 + h as u32);

            // Expected (whole plane, single tile).
            let mut expected = vec![0f32; spatial * patch_dim];
            cpu_build_im2col(&input, &mut expected, c_in, h, w, k, pad, h_out, w_out);

            // Device im2col over the whole plane.
            let d_input = graph.stream.clone_htod(&input).expect("htod im2col input");
            let mut d_patches = graph
                .stream
                .alloc_zeros::<f32>(spatial * patch_dim)
                .expect("alloc patches");
            unsafe {
                graph
                    .launch_imagen_vae_im2col(
                        &d_input,
                        &mut d_patches,
                        c_in as u32,
                        h as u32,
                        w as u32,
                        k as u32,
                        pad as u32,
                        w_out as u32,
                        0,
                        (spatial * patch_dim) as u32,
                    )
                    .expect("launch im2col");
            }
            graph.stream.synchronize().expect("sync im2col");
            let mut got = vec![0f32; spatial * patch_dim];
            graph
                .stream
                .memcpy_dtoh(&d_patches, &mut got)
                .expect("dtoh patches");
            graph.stream.synchronize().expect("sync dtoh");

            assert_eq!(
                got, expected,
                "im2col mismatch c_in={c_in} h={h} w={w} (k=3,pad=1)"
            );
        }
    }

    /// Parity: `encode_conv2d_f32` k=1 (pure channel-mix) vs CPU Conv2d.
    #[test]
    fn encode_conv2d_f32_k1_matches_cpu() {
        let _serial = super::super::types::gpu_parity_test_guard();
        let Some(graph) = graph_or_skip("conv2d k=1 parity") else {
            return;
        };
        let cases = [
            (3usize, 8usize, 5usize, 7usize),
            (16, 32, 12, 12),
            (35, 17, 10, 10), // odd channels
        ];
        let key_base: u64 = 9_100_000;
        for (idx, &(c_in, c_out, h, w)) in cases.iter().enumerate() {
            let input = vae_fill(c_in * h * w, c_in as u32 * 31 + h as u32);
            let weight = vae_fill(c_out * c_in, c_out as u32 * 17 + c_in as u32); // [c_out, c_in]
            let bias = vae_fill(c_out, c_out as u32 * 7 + 3);
            let handle = graph
                .get_or_upload_f32_weight(key_base + idx as u64, &weight)
                .expect("upload conv k1 weight");

            let mut got = vec![0f32; c_out * h * w];
            graph
                .encode_conv2d_f32(&handle, &input, &bias, &mut got, c_in, c_out, h, w, 1, 0)
                .expect("encode_conv2d_f32 k1");

            let (expected, ho, wo) = cpu_conv2d(&input, &weight, &bias, c_in, c_out, h, w, 1, 0);
            assert_eq!((ho, wo), (h, w));
            let (max_abs, rel_l2) = err_stats(&expected, &got);
            assert!(
                max_abs < 1e-3 && rel_l2 < 1e-4,
                "conv k1 C_in={c_in} C_out={c_out} H={h} W={w}: max_abs={max_abs:e} relL2={rel_l2:e}"
            );
        }
    }

    /// Parity: `encode_conv2d_f32` k=3 (GPU im2col + GEMM, tiled) vs CPU Conv2d.
    #[test]
    fn encode_conv2d_f32_k3_matches_cpu() {
        let _serial = super::super::types::gpu_parity_test_guard();
        let Some(graph) = graph_or_skip("conv2d k=3 parity") else {
            return;
        };
        let cases = [
            (4usize, 6usize, 6usize, 8usize),
            (16, 32, 12, 12),
            (35, 17, 10, 10), // odd channels
        ];
        let key_base: u64 = 9_200_000;
        for (idx, &(c_in, c_out, h, w)) in cases.iter().enumerate() {
            let patch_dim = 9 * c_in;
            let input = vae_fill(c_in * h * w, c_in as u32 * 53 + w as u32);
            let weight = vae_fill(c_out * patch_dim, c_out as u32 * 41 + c_in as u32); // [c_out, 3*3*c_in]
            let bias = vae_fill(c_out, c_out as u32 * 11 + 5);
            let handle = graph
                .get_or_upload_f32_weight(key_base + idx as u64, &weight)
                .expect("upload conv k3 weight");

            let mut got = vec![0f32; c_out * h * w];
            graph
                .encode_conv2d_f32(&handle, &input, &bias, &mut got, c_in, c_out, h, w, 3, 1)
                .expect("encode_conv2d_f32 k3");

            let (expected, ho, wo) = cpu_conv2d(&input, &weight, &bias, c_in, c_out, h, w, 3, 1);
            assert_eq!((ho, wo), (h, w)); // same padding
            let (max_abs, rel_l2) = err_stats(&expected, &got);
            assert!(
                max_abs < 1e-3 && rel_l2 < 1e-4,
                "conv k3 C_in={c_in} C_out={c_out} H={h} W={w}: max_abs={max_abs:e} relL2={rel_l2:e}"
            );
        }
    }

    /// Parity: `encode_groupnorm_f32` (f64 reduction) vs CPU GroupNorm (f64).
    #[test]
    fn encode_groupnorm_f32_matches_cpu() {
        let _serial = super::super::types::gpu_parity_test_guard();
        let Some(graph) = graph_or_skip("groupnorm parity") else {
            return;
        };
        let eps = 1e-6f32;
        let ng = 32usize;
        // (channels, hw) — channels divisible by 32, hw ∈ {16, 256}.
        let cases = [(32usize, 16usize), (64, 16), (32, 256), (64, 256)];
        for &(channels, hw) in &cases {
            // Non-identity affine, wider distribution so var is non-trivial.
            let base = vae_fill(channels * hw, channels as u32 * 23 + hw as u32);
            let mut x: Vec<f32> = base.iter().map(|v| v * 4.0 + 0.3).collect();
            let weight = vae_fill(channels, channels as u32 * 5 + 1)
                .iter()
                .map(|v| v + 1.0) // scale around 1
                .collect::<Vec<_>>();
            let bias = vae_fill(channels, channels as u32 * 3 + 2);

            let mut expected = x.clone();
            cpu_groupnorm(&mut expected, &weight, &bias, channels, hw, ng, eps);

            graph
                .encode_groupnorm_f32(&mut x, &weight, &bias, channels, hw, ng, eps)
                .expect("encode_groupnorm_f32");

            let (max_abs, rel_l2) = err_stats(&expected, &x);
            assert!(
                max_abs < 1e-4,
                "groupnorm C={channels} hw={hw}: max_abs={max_abs:e} relL2={rel_l2:e}"
            );
        }
    }

    /// Parity: `encode_silu_f32` (incl. negatives, non-256-multiple len) vs CPU.
    #[test]
    fn encode_silu_f32_matches_cpu() {
        let _serial = super::super::types::gpu_parity_test_guard();
        let Some(graph) = graph_or_skip("silu parity") else {
            return;
        };
        // len = 257 (non-multiple of 256), values span negatives via vae_fill
        // (already in [-0.5, 0.5)); widen so the sigmoid argument is non-trivial.
        let raw = vae_fill(257, 4242);
        let mut x: Vec<f32> = raw.iter().map(|v| v * 8.0).collect();
        let expected = cpu_silu(&x);

        graph.encode_silu_f32(&mut x).expect("encode_silu_f32");

        let (max_abs, rel_l2) = err_stats(&expected, &x);
        assert!(
            max_abs < 1e-4,
            "silu len=257: max_abs={max_abs:e} relL2={rel_l2:e}"
        );
    }

    /// Parity: `encode_upsample_nearest_f32` vs CPU — EXACT equality (gather).
    #[test]
    fn encode_upsample_nearest_f32_matches_cpu_exact() {
        let _serial = super::super::types::gpu_parity_test_guard();
        let Some(graph) = graph_or_skip("upsample parity") else {
            return;
        };
        let (c, h, w) = (3usize, 5usize, 5usize);
        let input = vae_fill(c * h * w, 7777);
        let expected = cpu_upsample_nearest(&input, c, h, w);

        let mut got = vec![0f32; c * 4 * h * w];
        graph
            .encode_upsample_nearest_f32(&input, &mut got, c, h, w)
            .expect("encode_upsample_nearest_f32");

        assert_eq!(got, expected, "upsample mismatch c={c} h={h} w={w}");
    }
}
