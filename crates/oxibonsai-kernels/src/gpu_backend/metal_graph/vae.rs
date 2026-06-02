//! Public `MetalGraph::encode_*` entry points for the **FLUX.2 VAE decoder**
//! per-op f32 GPU primitives.
//!
//! Each mirrors the `encode_gemm_f32` contract (upload → encode → commit → wait
//! → download) and is parity-validated against the CPU reference in
//! `oxibonsai-image::vae` (`conv.rs`, `norm.rs`, `ops.rs`):
//!
//! - [`MetalGraph::encode_conv2d_f32`] — stride-1 "same"-padded Conv2d. For
//!   `k=1, pad=0` it is a pure channel-mix that reuses the parity-clean
//!   [`encode_gemm_f32`](MetalGraph::encode_gemm_f32). For `k≥1, pad≥0` it runs a
//!   GPU `im2col_f32` (patches in `(kH,kW,C_in)` order, matching the MLX weight
//!   `[C_out,kH,kW,C_in]` flattening) **chained in one command buffer** with the
//!   same `gemm_f32_simdgroup` kernel, tiled over output rows so the im2col
//!   buffer stays bounded.
//! - [`MetalGraph::encode_groupnorm_f32`] — GroupNorm (32 groups, eps 1e-6).
//! - [`MetalGraph::encode_silu_f32`] — element-wise SiLU.
//! - [`MetalGraph::encode_upsample_nearest_f32`] — nearest ×2 upsample.
//!
//! All buffers are flat NCHW `[C, H, W]` f32 (batch 1). Resnet residual adds
//! reuse the existing `dispatch_residual_add` (no new kernel); the mid-block
//! attention stays on CPU for the first cut.

use metal::MTLResourceOptions;

use super::buffers::{alloc_buf, download_f32, upload_f32};
use super::error::{MetalGraphError, MetalWeightHandle};
use super::graph::MetalGraph;

/// Cap (bytes) on the per-tile im2col patch buffer for the k≥3 conv path.
///
/// The full im2col matrix for the largest VAE conv (up2, `[262144, 3456]`) is
/// ~3.6 GB, far too big to materialize at once. Output rows are tiled so the
/// patch buffer never exceeds this cap (~256 MiB), which for up2's `patch_dim =
/// 3456` admits ~19 k rows/tile (≈14 tiles) — large enough that the per-tile
/// command-buffer overhead is negligible against the GEMM.
const IM2COL_TILE_CAP_BYTES: usize = 256 * 1024 * 1024;

/// Whether the im2col-free implicit-GEMM conv path is disabled (forcing the
/// legacy tiled-im2col fallback). Off unless `OXI_VAE_NO_IMPLICIT_CONV=1`. The
/// env read is cached on first call. Diagnostic only (A/B timing / safety hatch).
fn implicit_conv_disabled() -> bool {
    use std::sync::OnceLock;
    static DISABLED: OnceLock<bool> = OnceLock::new();
    *DISABLED.get_or_init(|| {
        matches!(
            std::env::var("OXI_VAE_NO_IMPLICIT_CONV").ok().as_deref(),
            Some("1")
        )
    })
}

impl MetalGraph {
    /// Stride-1, "same"-padded 2-D convolution on an NCHW input `[C_in, H, W]`
    /// (batch 1), returning the NCHW output `[C_out, H_out, W_out]` (`H_out = H +
    /// 2·pad − k + 1`), with the MLX-layout weight `[C_out, kH, kW, C_in]`
    /// (flattened row-major == `[C_out, kH·kW·C_in]`) and a per-output-channel
    /// `bias[C_out]`.
    ///
    /// # Algorithm
    ///
    /// - **`k == 1 && pad == 0`** — a pure channel-mix: reshape the input
    ///   `[C_in, H·W]` → `[H·W, C_in]`, run the parity-clean
    ///   [`encode_gemm_f32`](MetalGraph::encode_gemm_f32) (`m = H·W`, `n =
    ///   C_out`, `k = C_in`), then transpose `[H·W, C_out]` → `[C_out, H·W]` and
    ///   add the bias.
    /// - **otherwise (any `k ≥ 1`, any `pad ≥ 0`)** — GPU `im2col_f32` produces
    ///   patches `[rows, kH·kW·C_in]` in the `(kH,kW,C_in)` order that matches
    ///   the weight flattening, **chained in a single command buffer** with the
    ///   `gemm_f32_simdgroup` GEMM (same kernel `encode_gemm_f32` uses). Output
    ///   rows are tiled so the (GPU-private) patch buffer stays ≤
    ///   `IM2COL_TILE_CAP_BYTES`; each tile's `[rows, C_out]` result is
    ///   downloaded and scattered into the NCHW output with the bias.
    ///
    /// The element order is identical to `oxibonsai-image::vae::conv` (im2col +
    /// `gemm_abt`), so the result is bit-compatible up to f32 sum reassociation.
    ///
    /// # Parameters
    /// - `weight`: pre-uploaded row-major f32 weight handle, `[C_out, kH·kW·C_in]`
    ///   (e.g. via [`get_or_upload_f32_weight`](MetalGraph::get_or_upload_f32_weight)).
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
    /// [`MetalGraphError::InvalidDimensions`] on any length / shape mismatch (or
    /// degenerate `h_out`/`w_out` ≤ 0), or a buffer / execution error.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_conv2d_f32(
        &self,
        weight: &MetalWeightHandle,
        input: &[f32],
        bias: &[f32],
        output: &mut [f32],
        c_in: usize,
        c_out: usize,
        h: usize,
        w: usize,
        k: usize,
        pad: usize,
    ) -> Result<(), MetalGraphError> {
        // ── Validate ─────────────────────────────────────────────────────
        if k == 0 {
            return Err(MetalGraphError::InvalidDimensions(
                "encode_conv2d_f32: kernel size k must be ≥ 1".into(),
            ));
        }
        let expected_in = c_in
            .checked_mul(h)
            .and_then(|x| x.checked_mul(w))
            .ok_or_else(|| {
                MetalGraphError::InvalidDimensions(format!(
                    "encode_conv2d_f32: c_in*h*w overflow (c_in={c_in}, h={h}, w={w})"
                ))
            })?;
        if input.len() != expected_in {
            return Err(MetalGraphError::InvalidDimensions(format!(
                "encode_conv2d_f32: input len {} != c_in*h*w {expected_in}",
                input.len()
            )));
        }
        if bias.len() != c_out {
            return Err(MetalGraphError::InvalidDimensions(format!(
                "encode_conv2d_f32: bias len {} != c_out {c_out}",
                bias.len()
            )));
        }
        // "same"-stride-1 output geometry: h_out = h + 2*pad - k + 1.
        let h_pad = h + 2 * pad;
        let w_pad = w + 2 * pad;
        if h_pad + 1 < k || w_pad + 1 < k {
            return Err(MetalGraphError::InvalidDimensions(format!(
                "encode_conv2d_f32: kernel {k} larger than padded input {h_pad}x{w_pad}"
            )));
        }
        let h_out = h_pad + 1 - k;
        let w_out = w_pad + 1 - k;
        let spatial = h_out.checked_mul(w_out).ok_or_else(|| {
            MetalGraphError::InvalidDimensions(format!(
                "encode_conv2d_f32: h_out*w_out overflow (h_out={h_out}, w_out={w_out})"
            ))
        })?;
        let expected_out = c_out.checked_mul(spatial).ok_or_else(|| {
            MetalGraphError::InvalidDimensions(format!(
                "encode_conv2d_f32: c_out*spatial overflow (c_out={c_out}, spatial={spatial})"
            ))
        })?;
        if output.len() != expected_out {
            return Err(MetalGraphError::InvalidDimensions(format!(
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

        // ── k ≥ 1, pad ≥ 0: contraction dim and weight-size validation (shared
        // by the implicit-GEMM fast path and the tiled-im2col fallback). ──
        let patch_dim = k
            .checked_mul(k)
            .and_then(|x| x.checked_mul(c_in))
            .ok_or_else(|| {
                MetalGraphError::InvalidDimensions(format!(
                    "encode_conv2d_f32: patch_dim overflow (k={k}, c_in={c_in})"
                ))
            })?;
        // Verify the weight buffer is large enough for [c_out, patch_dim] f32.
        let weight_floats = c_out.checked_mul(patch_dim).ok_or_else(|| {
            MetalGraphError::InvalidDimensions(format!(
                "encode_conv2d_f32: c_out*patch_dim overflow (c_out={c_out}, patch_dim={patch_dim})"
            ))
        })?;
        let weight_bytes = weight_floats
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                MetalGraphError::InvalidDimensions(
                    "encode_conv2d_f32: weight byte size overflow".into(),
                )
            })?;
        if weight.byte_len < weight_bytes {
            return Err(MetalGraphError::InvalidDimensions(format!(
                "encode_conv2d_f32: weight handle holds {} bytes < c_out*patch_dim*4 {weight_bytes}",
                weight.byte_len
            )));
        }

        // ── Preferred path: im2col-FREE implicit-GEMM conv (the high-res VAE
        // convs). Gathers conv patches on-the-fly into threadgroup memory and
        // drives `simdgroup_float8x8` HW MACs, so the ~GB im2col patch matrix
        // never hits global memory (≈2× the GFLOP/s of the materialized path).
        // The kernel indexing is fully general, but we route only the convs it is
        // tuned for (k ≥ 3, pad ≥ 1) here and KEEP the tiled-im2col path below as
        // the fallback for everything else (and on any encode error).
        //
        // Diagnostic kill-switch: `OXI_VAE_NO_IMPLICIT_CONV=1` forces the legacy
        // tiled-im2col path (for A/B timing, or as a safety hatch on hardware the
        // implicit kernel hasn't been validated on). Default OFF. ──
        if k >= 3 && pad >= 1 && !implicit_conv_disabled() {
            match self.encode_conv2d_f32_implicit(
                weight, input, bias, output, c_in, c_out, h, w, k, pad, spatial, patch_dim, w_out,
            ) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    // Never fail the decode: fall through to the im2col path.
                    tracing::debug!("conv2d implicit-GEMM path failed ({e}); im2col fallback");
                }
            }
        }

        // ── Fallback path: GPU im2col (tiled) chained with gemm_f32. ──
        self.encode_conv2d_f32_im2col(
            weight, input, bias, output, c_in, c_out, h, w, k, pad, spatial, patch_dim, w_out,
        )
    }

    /// Tiled-im2col conv2d fallback: GPU `im2col_f32` (patches materialized to a
    /// capped GPU-private buffer) chained in one command buffer with
    /// `gemm_f32_simdgroup`, output rows tiled so the patch buffer stays ≤
    /// [`IM2COL_TILE_CAP_BYTES`]. Retained behind the implicit-GEMM fast path so
    /// nothing regresses for shapes that path does not cover (and as the
    /// on-error fallback). Preconditions match
    /// [`encode_conv2d_f32_implicit`](Self::encode_conv2d_f32_implicit).
    ///
    /// `pub(crate)` so the in-crate parity/speed tests can drive the im2col path
    /// directly (A/B against the implicit kernel); not part of the public API.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_conv2d_f32_im2col(
        &self,
        weight: &MetalWeightHandle,
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
    ) -> Result<(), MetalGraphError> {
        let shared = MTLResourceOptions::StorageModeShared;
        let private = MTLResourceOptions::StorageModePrivate;

        // Upload the NCHW input once (shared so im2col reads it directly).
        let input_bytes = std::mem::size_of_val(input) as u64;
        let input_buf = alloc_buf(&self.device, input_bytes, shared)?;
        unsafe { upload_f32(&input_buf, input) };

        // Tile the output rows so the patch buffer stays ≤ the cap.
        let patch_row_bytes = patch_dim
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                MetalGraphError::InvalidDimensions(
                    "encode_conv2d_f32: patch row byte size overflow".into(),
                )
            })?;
        let tile_rows = (IM2COL_TILE_CAP_BYTES / patch_row_bytes.max(1))
            .max(1)
            .min(spatial);

        // GPU-private patch scratch [tile_rows, patch_dim]; shared output tile
        // [tile_rows, c_out] for download.
        let patches_bytes = (tile_rows * patch_dim * std::mem::size_of::<f32>()) as u64;
        let patches_buf = alloc_buf(&self.device, patches_bytes, private)?;
        let out_tile_bytes = (tile_rows * c_out * std::mem::size_of::<f32>()) as u64;
        let out_tile_buf = alloc_buf(&self.device, out_tile_bytes, shared)?;

        let mut out_tile = vec![0f32; tile_rows * c_out];

        let mut row_start = 0usize;
        while row_start < spatial {
            let rows = (spatial - row_start).min(tile_rows);
            let n_elems = rows * patch_dim;

            let cmd_buf = self.command_queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();

            // im2col → patches [rows, patch_dim] (GPU-private).
            self.dispatch_im2col_f32(
                encoder,
                &input_buf,
                &patches_buf,
                c_in as u32,
                h as u32,
                w as u32,
                k as u32,
                pad as u32,
                w_out as u32,
                row_start as u32,
                n_elems as u32,
            );
            // GEMM: out_tile[rows, c_out] = patches[rows, patch_dim] · weightᵀ.
            // Same buffer-(0/1/2) contract as encode_gemm_f32's dispatch; the
            // RAW dependency on `patches_buf` is enforced by Metal's automatic
            // hazard tracking (single non-concurrent encoder).
            self.dispatch_gemm_f32(
                encoder,
                &weight.buffer,
                &patches_buf,
                &out_tile_buf,
                c_out as u32,
                patch_dim as u32,
                rows as u32,
            );

            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            // Download this tile and scatter into NCHW output with bias.
            // out_tile is row-major [rows, c_out] (outputs[m*c_out + oc]).
            unsafe { download_f32(&out_tile_buf, &mut out_tile[..rows * c_out]) };
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

    /// **im2col-free implicit-GEMM** conv2d (the high-res VAE convs): dispatches
    /// the `conv2d_f32_implicit` kernel, which gathers conv patches on-the-fly
    /// into threadgroup memory (never materializing the im2col matrix to global
    /// memory) and drives `simdgroup_float8x8` HW MACs. Computes the same op as
    /// the tiled-im2col path — `out[C_out, P] = weight[C_out, patch_dim] ·
    /// Patches[patch_dim, P]` — then adds the per-channel bias on download, so the
    /// NCHW `[C_out, H_out, W_out]` result is bit-compatible up to f32 sum
    /// reassociation.
    ///
    /// Preconditions (validated by the caller [`encode_conv2d_f32`]): `spatial =
    /// h_out·w_out`, `patch_dim = k·k·c_in`, the weight handle holds ≥
    /// `c_out·patch_dim` f32, and `input`/`output` lengths are `c_in·h·w` /
    /// `c_out·spatial`. The kernel indexing is general in `k`/`pad`, so this also
    /// serves any `k ≥ 1`.
    ///
    /// On any allocation/encode error the caller falls back to the im2col path, so
    /// this is purely an accelerator.
    ///
    /// `pub(crate)` so the in-crate parity/speed tests can drive the implicit
    /// kernel directly (independent of the `encode_conv2d_f32` routing); not part
    /// of the public API.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_conv2d_f32_implicit(
        &self,
        weight: &MetalWeightHandle,
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
    ) -> Result<(), MetalGraphError> {
        let shared = MTLResourceOptions::StorageModeShared;

        // Upload the NCHW input once (shared so the gather reads it directly).
        let input_bytes = std::mem::size_of_val(input) as u64;
        let input_buf = alloc_buf(&self.device, input_bytes, shared)?;
        unsafe { upload_f32(&input_buf, input) };

        // Shared output buffer [c_out, spatial] (row-major NCHW; NO bias yet).
        let out_floats = c_out.checked_mul(spatial).ok_or_else(|| {
            MetalGraphError::InvalidDimensions(format!(
                "encode_conv2d_f32_implicit: c_out*spatial overflow (c_out={c_out}, spatial={spatial})"
            ))
        })?;
        let out_bytes = out_floats
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                MetalGraphError::InvalidDimensions(
                    "encode_conv2d_f32_implicit: output byte size overflow".into(),
                )
            })? as u64;
        let out_buf = alloc_buf(&self.device, out_bytes, shared)?;

        // Single command buffer: one implicit-GEMM conv over the whole plane.
        let cmd_buf = self.command_queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        self.dispatch_conv2d_f32_implicit(
            encoder,
            &weight.buffer,
            &input_buf,
            &out_buf,
            c_out as u32,
            spatial as u32,
            patch_dim as u32,
            c_in as u32,
            h as u32,
            w as u32,
            k as u32,
            pad as u32,
            w_out as u32,
        );
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        // Download row-major [c_out, spatial] and add the per-channel bias in
        // place (the kernel intentionally omits it, matching the im2col path's
        // CPU-side bias add).
        unsafe { download_f32(&out_buf, output) };
        for oc in 0..c_out {
            let b = bias[oc];
            let dst = &mut output[oc * spatial..(oc + 1) * spatial];
            for slot in dst.iter_mut() {
                *slot += b;
            }
        }

        Ok(())
    }

    /// PyTorch-compatible GroupNorm on an NCHW buffer `[C, H, W]` (batch 1), in
    /// place: split `C` into `num_groups` contiguous groups, normalize each over
    /// all its channels × spatial positions (`(x − mean) / sqrt(var + eps)`,
    /// population variance), then apply the per-channel affine `weight[c]` /
    /// `bias[c]`. Mirrors `oxibonsai-image::vae::norm::forward_inplace`.
    ///
    /// One threadgroup per group performs a Kahan-compensated f32 reduction (the
    /// CPU reference accumulates in f64; Metal has no `double`, and the
    /// compensated f32 sum stays within ≪ 1e-4 of f64 over the VAE group sizes —
    /// see `MSL_GROUPNORM_F32`).
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
    /// [`MetalGraphError::InvalidDimensions`] on a length / divisibility
    /// mismatch, or a buffer / execution error.
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
    ) -> Result<(), MetalGraphError> {
        if num_groups == 0 || channels % num_groups != 0 {
            return Err(MetalGraphError::InvalidDimensions(format!(
                "encode_groupnorm_f32: channels {channels} not divisible by num_groups {num_groups}"
            )));
        }
        let expected = channels.checked_mul(hw).ok_or_else(|| {
            MetalGraphError::InvalidDimensions(format!(
                "encode_groupnorm_f32: channels*hw overflow (channels={channels}, hw={hw})"
            ))
        })?;
        if x.len() != expected {
            return Err(MetalGraphError::InvalidDimensions(format!(
                "encode_groupnorm_f32: x len {} != channels*hw {expected}",
                x.len()
            )));
        }
        if weight.len() != channels || bias.len() != channels {
            return Err(MetalGraphError::InvalidDimensions(format!(
                "encode_groupnorm_f32: weight/bias len ({}/{}) != channels {channels}",
                weight.len(),
                bias.len()
            )));
        }
        if expected == 0 {
            return Ok(());
        }

        let shared = MTLResourceOptions::StorageModeShared;
        let x_bytes = std::mem::size_of_val(&x[..]) as u64;
        let aff_bytes = (channels * std::mem::size_of::<f32>()) as u64;
        let x_buf = alloc_buf(&self.device, x_bytes, shared)?;
        let w_buf = alloc_buf(&self.device, aff_bytes, shared)?;
        let b_buf = alloc_buf(&self.device, aff_bytes, shared)?;
        unsafe {
            upload_f32(&x_buf, x);
            upload_f32(&w_buf, weight);
            upload_f32(&b_buf, bias);
        }

        let cmd_buf = self.command_queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        self.dispatch_groupnorm_f32(
            encoder,
            &x_buf,
            &w_buf,
            &b_buf,
            channels as u32,
            hw as u32,
            num_groups as u32,
            eps,
        );
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        unsafe { download_f32(&x_buf, x) };
        Ok(())
    }

    /// Element-wise SiLU (`x · sigmoid(x) = x / (1 + exp(-x))`), in place over a
    /// flat f32 buffer. Mirrors `oxibonsai-image::math::silu`.
    ///
    /// # Errors
    /// A buffer / execution error if the GPU work cannot be encoded.
    pub fn encode_silu_f32(&self, x: &mut [f32]) -> Result<(), MetalGraphError> {
        if x.is_empty() {
            return Ok(());
        }
        let shared = MTLResourceOptions::StorageModeShared;
        let x_bytes = std::mem::size_of_val(&x[..]) as u64;
        let x_buf = alloc_buf(&self.device, x_bytes, shared)?;
        unsafe { upload_f32(&x_buf, x) };

        let cmd_buf = self.command_queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        self.dispatch_silu_f32(encoder, &x_buf, x.len() as u32);
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        unsafe { download_f32(&x_buf, x) };
        Ok(())
    }

    /// Nearest-neighbour ×2 upsample of an NCHW buffer `[C, H, W]` → `[C, 2H,
    /// 2W]` (each pixel repeated 2× along H and W). Mirrors
    /// `oxibonsai-image::vae::ops::upsample_nearest2x`.
    ///
    /// # Parameters
    /// - `input`: NCHW `[C, H·W]`, length `c * h * w`.
    /// - `output`: NCHW `[C, 2H·2W]` (overwritten), length `c * 4 * h * w`.
    /// - `c` / `h` / `w`: input channel / spatial dims.
    ///
    /// # Errors
    /// [`MetalGraphError::InvalidDimensions`] on a length mismatch, or a buffer /
    /// execution error.
    pub fn encode_upsample_nearest_f32(
        &self,
        input: &[f32],
        output: &mut [f32],
        c: usize,
        h: usize,
        w: usize,
    ) -> Result<(), MetalGraphError> {
        let expected_in = c
            .checked_mul(h)
            .and_then(|x| x.checked_mul(w))
            .ok_or_else(|| {
                MetalGraphError::InvalidDimensions(format!(
                    "encode_upsample_nearest_f32: c*h*w overflow (c={c}, h={h}, w={w})"
                ))
            })?;
        if input.len() != expected_in {
            return Err(MetalGraphError::InvalidDimensions(format!(
                "encode_upsample_nearest_f32: input len {} != c*h*w {expected_in}",
                input.len()
            )));
        }
        let expected_out = expected_in.checked_mul(4).ok_or_else(|| {
            MetalGraphError::InvalidDimensions(
                "encode_upsample_nearest_f32: output size overflow".into(),
            )
        })?;
        if output.len() != expected_out {
            return Err(MetalGraphError::InvalidDimensions(format!(
                "encode_upsample_nearest_f32: output len {} != c*4*h*w {expected_out}",
                output.len()
            )));
        }
        if expected_in == 0 {
            return Ok(());
        }

        let shared = MTLResourceOptions::StorageModeShared;
        let in_bytes = std::mem::size_of_val(input) as u64;
        let out_bytes = std::mem::size_of_val(&output[..]) as u64;
        let in_buf = alloc_buf(&self.device, in_bytes, shared)?;
        let out_buf = alloc_buf(&self.device, out_bytes, shared)?;
        unsafe { upload_f32(&in_buf, input) };

        let cmd_buf = self.command_queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        self.dispatch_upsample_nearest_f32(
            encoder,
            &in_buf,
            &out_buf,
            c as u32,
            h as u32,
            w as u32,
            expected_out as u32,
        );
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        unsafe { download_f32(&out_buf, output) };
        Ok(())
    }
}
