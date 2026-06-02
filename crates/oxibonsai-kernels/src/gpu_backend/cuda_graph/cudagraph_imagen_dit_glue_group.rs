//! # CudaGraph — FLUX.2 DiT glue-op encode methods
//!
//! Host-side launchers + public `encode_*` entry points for the six DiT glue
//! kernels in [`CUDA_IMAGEN_DIT_GLUE_SRC`](super::super::cuda_imagen_dit_glue_kernels::CUDA_IMAGEN_DIT_GLUE_SRC):
//! `modulate_f32`, `gated_residual_add_f32`, `layer_norm_f32`,
//! `rms_norm_heads_f32`, `swiglu_f32`, `rope_interleaved_f32`.
//!
//! Each op exposes:
//! - an `unsafe fn launch_dit_*` device-level launcher (operates on
//!   `CudaSlice<f32>` device buffers — the building block the **resident** DiT
//!   forward (Phase 2) chains without host round-trips, mirroring
//!   `launch_gemm_f32`); and
//! - a public host-API `encode_dit_*` (upload → launch → download) that is the
//!   parity unit-test surface and a drop-in for the per-op CPU references in
//!   `oxibonsai-image::math` / `::blocks`.
//!
//! Parity-first plain FP32 (double-accumulated reductions); validated at
//! cos ≥ 0.999 against the CPU ports.

#![cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use super::types::CudaGraphError;

use super::cudagraph_type::CudaGraph;

/// Threads per block for the element-wise grid-stride kernels.
const DIT_GLUE_BLOCK: u32 = 256;
/// Cap on grid blocks for the element-wise kernels (grid-stride covers the rest).
const DIT_GLUE_MAX_GRID: u32 = 4096;

/// `grid` for an element-wise kernel over `n` items: `ceil(n/block)`, capped.
fn elementwise_grid(n: usize) -> u32 {
    let blocks = (n as u64).div_ceil(DIT_GLUE_BLOCK as u64);
    blocks.clamp(1, DIT_GLUE_MAX_GRID as u64) as u32
}

impl CudaGraph {
    // ── modulate ───────────────────────────────────────────────────────────

    /// Launch `modulate_f32` (`x = (1+scale)*x + shift`, in place) on `self.stream`.
    ///
    /// # Safety
    /// `d_x` holds ≥ `rows*dim` f32; `d_shift`/`d_scale` hold ≥ `dim` f32; all on
    /// `self.stream`.
    pub(crate) unsafe fn launch_dit_modulate(
        &self,
        d_x: &mut CudaSlice<f32>,
        d_shift: &CudaSlice<f32>,
        d_scale: &CudaSlice<f32>,
        rows: u32,
        dim: u32,
    ) -> Result<(), CudaGraphError> {
        let cfg = LaunchConfig {
            grid_dim: (elementwise_grid(rows as usize * dim as usize), 1, 1),
            block_dim: (DIT_GLUE_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        self.stream
            .launch_builder(&self.modules.dit_modulate)
            .arg(d_x)
            .arg(d_shift)
            .arg(d_scale)
            .arg(&rows)
            .arg(&dim)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("modulate_f32 launch: {e}")))
    }

    /// Modulate `x` `[rows, dim]` in place: `x = (1 + scale) * x + shift`
    /// (`scale`/`shift` length `dim`). Host-API wrapper (mirrors
    /// `oxibonsai-image::math::modulate_inplace`).
    ///
    /// # Errors
    /// [`CudaGraphError::DriverError`] on a length mismatch or buffer/launch error.
    pub fn encode_dit_modulate(
        &self,
        x: &mut [f32],
        shift: &[f32],
        scale: &[f32],
        rows: usize,
        dim: usize,
    ) -> Result<(), CudaGraphError> {
        let expected = rows.checked_mul(dim).ok_or_else(|| {
            CudaGraphError::DriverError("encode_dit_modulate: rows*dim overflow".into())
        })?;
        if x.len() != expected || shift.len() != dim || scale.len() != dim {
            return Err(CudaGraphError::DriverError(format!(
                "encode_dit_modulate: bad lens (x {} != {expected}, shift {} / scale {} != dim {dim})",
                x.len(),
                shift.len(),
                scale.len()
            )));
        }
        if expected == 0 {
            return Ok(());
        }
        let mut d_x = self.htod(x)?;
        let d_shift = self.htod(shift)?;
        let d_scale = self.htod(scale)?;
        unsafe {
            self.launch_dit_modulate(&mut d_x, &d_shift, &d_scale, rows as u32, dim as u32)?;
        }
        self.dtoh_sync(&d_x, x)
    }

    // ── gated residual add ───────────────────────────────────────────────────

    /// Launch `gated_residual_add_f32` (`h += gate*delta`, in place).
    ///
    /// # Safety
    /// `d_h`/`d_delta` hold ≥ `rows*dim` f32; `d_gate` ≥ `dim`; all on `self.stream`.
    pub(crate) unsafe fn launch_dit_gated_residual_add(
        &self,
        d_h: &mut CudaSlice<f32>,
        d_delta: &CudaSlice<f32>,
        d_gate: &CudaSlice<f32>,
        rows: u32,
        dim: u32,
    ) -> Result<(), CudaGraphError> {
        let cfg = LaunchConfig {
            grid_dim: (elementwise_grid(rows as usize * dim as usize), 1, 1),
            block_dim: (DIT_GLUE_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        self.stream
            .launch_builder(&self.modules.dit_gated_residual_add)
            .arg(d_h)
            .arg(d_delta)
            .arg(d_gate)
            .arg(&rows)
            .arg(&dim)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("gated_residual_add_f32 launch: {e}")))
    }

    /// `h += gate * delta` over `[rows, dim]` (`gate` length `dim`), in place.
    /// Mirrors `oxibonsai-image::blocks::gated_residual_add`.
    ///
    /// # Errors
    /// [`CudaGraphError::DriverError`] on a length mismatch or buffer/launch error.
    pub fn encode_dit_gated_residual_add(
        &self,
        h: &mut [f32],
        delta: &[f32],
        gate: &[f32],
        rows: usize,
        dim: usize,
    ) -> Result<(), CudaGraphError> {
        let expected = rows.checked_mul(dim).ok_or_else(|| {
            CudaGraphError::DriverError("encode_dit_gated_residual_add: rows*dim overflow".into())
        })?;
        if h.len() != expected || delta.len() != expected || gate.len() != dim {
            return Err(CudaGraphError::DriverError(format!(
                "encode_dit_gated_residual_add: bad lens (h {} / delta {} != {expected}, gate {} != dim {dim})",
                h.len(),
                delta.len(),
                gate.len()
            )));
        }
        if expected == 0 {
            return Ok(());
        }
        let mut d_h = self.htod(h)?;
        let d_delta = self.htod(delta)?;
        let d_gate = self.htod(gate)?;
        unsafe {
            self.launch_dit_gated_residual_add(
                &mut d_h,
                &d_delta,
                &d_gate,
                rows as u32,
                dim as u32,
            )?;
        }
        self.dtoh_sync(&d_h, h)
    }

    // ── layer norm (affine = false) ──────────────────────────────────────────

    /// Launch `layer_norm_f32` (in place, one block per row).
    ///
    /// # Safety
    /// `d_x` holds ≥ `rows*dim` f32 on `self.stream`.
    pub(crate) unsafe fn launch_dit_layer_norm(
        &self,
        d_x: &mut CudaSlice<f32>,
        rows: u32,
        dim: u32,
        eps: f32,
    ) -> Result<(), CudaGraphError> {
        let cfg = LaunchConfig {
            grid_dim: (rows.max(1), 1, 1),
            block_dim: (256, 1, 1), // == LN_THREADS
            shared_mem_bytes: 0,
        };
        self.stream
            .launch_builder(&self.modules.dit_layer_norm)
            .arg(d_x)
            .arg(&rows)
            .arg(&dim)
            .arg(&eps)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("layer_norm_f32 launch: {e}")))
    }

    /// LayerNorm (affine = false) of `x` `[rows, dim]` in place: per row,
    /// `(x - mean) / sqrt(var + eps)` (population variance). Mirrors
    /// `oxibonsai-image::math::layer_norm_inplace`.
    ///
    /// # Errors
    /// [`CudaGraphError::DriverError`] on a length mismatch or buffer/launch error.
    pub fn encode_dit_layer_norm(
        &self,
        x: &mut [f32],
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<(), CudaGraphError> {
        let expected = rows.checked_mul(dim).ok_or_else(|| {
            CudaGraphError::DriverError("encode_dit_layer_norm: rows*dim overflow".into())
        })?;
        if x.len() != expected {
            return Err(CudaGraphError::DriverError(format!(
                "encode_dit_layer_norm: x len {} != rows*dim {expected}",
                x.len()
            )));
        }
        if expected == 0 || dim == 0 {
            return Ok(());
        }
        let mut d_x = self.htod(x)?;
        unsafe {
            self.launch_dit_layer_norm(&mut d_x, rows as u32, dim as u32, eps)?;
        }
        self.dtoh_sync(&d_x, x)
    }

    // ── per-head RMS norm (QK-RMSNorm) ───────────────────────────────────────

    /// Launch `rms_norm_heads_f32` (in place, one block per `head_dim` chunk).
    ///
    /// # Safety
    /// `d_x` holds ≥ `rows*head_dim` f32; `d_weight` ≥ `head_dim`; on `self.stream`.
    pub(crate) unsafe fn launch_dit_rms_norm_heads(
        &self,
        d_x: &mut CudaSlice<f32>,
        d_weight: &CudaSlice<f32>,
        rows: u32,
        head_dim: u32,
        eps: f32,
    ) -> Result<(), CudaGraphError> {
        let cfg = LaunchConfig {
            grid_dim: (rows.max(1), 1, 1),
            block_dim: (128, 1, 1), // == RMS_THREADS
            shared_mem_bytes: 0,
        };
        self.stream
            .launch_builder(&self.modules.dit_rms_norm_heads)
            .arg(d_x)
            .arg(d_weight)
            .arg(&rows)
            .arg(&head_dim)
            .arg(&eps)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("rms_norm_heads_f32 launch: {e}")))
    }

    /// Per-head QK-RMSNorm of `x` `[rows, head_dim]` (`rows = num_heads*seq`):
    /// `weight * x / sqrt(mean(x^2) + eps)` per `head_dim` chunk. In place.
    /// Mirrors `oxibonsai-image::math::rms_norm_heads_inplace`.
    ///
    /// # Errors
    /// [`CudaGraphError::DriverError`] on a length mismatch or buffer/launch error.
    pub fn encode_dit_rms_norm_heads(
        &self,
        x: &mut [f32],
        weight: &[f32],
        rows: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<(), CudaGraphError> {
        let expected = rows.checked_mul(head_dim).ok_or_else(|| {
            CudaGraphError::DriverError("encode_dit_rms_norm_heads: rows*head_dim overflow".into())
        })?;
        if x.len() != expected || weight.len() != head_dim {
            return Err(CudaGraphError::DriverError(format!(
                "encode_dit_rms_norm_heads: bad lens (x {} != {expected}, weight {} != head_dim {head_dim})",
                x.len(),
                weight.len()
            )));
        }
        if expected == 0 || head_dim == 0 {
            return Ok(());
        }
        let mut d_x = self.htod(x)?;
        let d_w = self.htod(weight)?;
        unsafe {
            self.launch_dit_rms_norm_heads(&mut d_x, &d_w, rows as u32, head_dim as u32, eps)?;
        }
        self.dtoh_sync(&d_x, x)
    }

    // ── SwiGLU ───────────────────────────────────────────────────────────────

    /// Launch `swiglu_f32`: `out[rows, half] = silu(x[:, :half]) * x[:, half:]`.
    ///
    /// # Safety
    /// `d_x` holds ≥ `rows*2*half` f32; `d_out` ≥ `rows*half`; on `self.stream`.
    pub(crate) unsafe fn launch_dit_swiglu(
        &self,
        d_x: &CudaSlice<f32>,
        d_out: &mut CudaSlice<f32>,
        rows: u32,
        half: u32,
    ) -> Result<(), CudaGraphError> {
        let cfg = LaunchConfig {
            grid_dim: (elementwise_grid(rows as usize * half as usize), 1, 1),
            block_dim: (DIT_GLUE_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        self.stream
            .launch_builder(&self.modules.dit_swiglu)
            .arg(d_x)
            .arg(d_out)
            .arg(&rows)
            .arg(&half)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("swiglu_f32 launch: {e}")))
    }

    /// SwiGLU over `x` `[rows, 2*half]` → `[rows, half]` = `silu(gate) * up`.
    /// Mirrors `oxibonsai-image::math::swiglu`.
    ///
    /// # Errors
    /// [`CudaGraphError::DriverError`] on a length mismatch or buffer/launch error.
    pub fn encode_dit_swiglu(
        &self,
        x: &[f32],
        out: &mut [f32],
        rows: usize,
        half: usize,
    ) -> Result<(), CudaGraphError> {
        let in_len = rows
            .checked_mul(2)
            .and_then(|x| x.checked_mul(half))
            .ok_or_else(|| {
                CudaGraphError::DriverError("encode_dit_swiglu: rows*2*half overflow".into())
            })?;
        let out_len = rows.checked_mul(half).ok_or_else(|| {
            CudaGraphError::DriverError("encode_dit_swiglu: rows*half overflow".into())
        })?;
        if x.len() != in_len || out.len() != out_len {
            return Err(CudaGraphError::DriverError(format!(
                "encode_dit_swiglu: bad lens (x {} != {in_len}, out {} != {out_len})",
                x.len(),
                out.len()
            )));
        }
        if out_len == 0 {
            return Ok(());
        }
        let d_x = self.htod(x)?;
        let mut d_out = self.alloc_zeros(out_len)?;
        unsafe {
            self.launch_dit_swiglu(&d_x, &mut d_out, rows as u32, half as u32)?;
        }
        self.dtoh_sync(&d_out, out)
    }

    // ── interleaved RoPE ─────────────────────────────────────────────────────

    /// Launch `rope_interleaved_f32` (in place) over head-major
    /// `x[num_heads, seq, head_dim]` with `cost`/`sint` `[seq, head_dim/2]`.
    ///
    /// # Safety
    /// `d_x` holds ≥ `num_heads*seq*head_dim` f32; `d_cos`/`d_sin` ≥
    /// `seq*head_dim/2`; on `self.stream`.
    pub(crate) unsafe fn launch_dit_rope(
        &self,
        d_x: &mut CudaSlice<f32>,
        d_cos: &CudaSlice<f32>,
        d_sin: &CudaSlice<f32>,
        num_heads: u32,
        seq: u32,
        head_dim: u32,
    ) -> Result<(), CudaGraphError> {
        let pairs = num_heads as usize * seq as usize * (head_dim as usize / 2);
        let cfg = LaunchConfig {
            grid_dim: (elementwise_grid(pairs), 1, 1),
            block_dim: (DIT_GLUE_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        self.stream
            .launch_builder(&self.modules.dit_rope_interleaved)
            .arg(d_x)
            .arg(d_cos)
            .arg(d_sin)
            .arg(&num_heads)
            .arg(&seq)
            .arg(&head_dim)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("rope_interleaved_f32 launch: {e}")))
    }

    /// Interleaved RoPE on head-major `x[num_heads, seq, head_dim]` in place;
    /// `cos`/`sin` are `[seq, head_dim/2]`. Mirrors
    /// `oxibonsai-image::math::apply_rope_inplace`.
    ///
    /// # Errors
    /// [`CudaGraphError::DriverError`] on a length/parity mismatch or buffer/launch
    /// error.
    pub fn encode_dit_rope(
        &self,
        x: &mut [f32],
        cos: &[f32],
        sin: &[f32],
        num_heads: usize,
        seq: usize,
        head_dim: usize,
    ) -> Result<(), CudaGraphError> {
        if head_dim % 2 != 0 {
            return Err(CudaGraphError::DriverError(format!(
                "encode_dit_rope: head_dim {head_dim} must be even"
            )));
        }
        let half = head_dim / 2;
        let expected = num_heads
            .checked_mul(seq)
            .and_then(|x| x.checked_mul(head_dim))
            .ok_or_else(|| CudaGraphError::DriverError("encode_dit_rope: x len overflow".into()))?;
        let tab = seq.checked_mul(half).ok_or_else(|| {
            CudaGraphError::DriverError("encode_dit_rope: seq*half overflow".into())
        })?;
        if x.len() != expected || cos.len() != tab || sin.len() != tab {
            return Err(CudaGraphError::DriverError(format!(
                "encode_dit_rope: bad lens (x {} != {expected}, cos {} / sin {} != seq*half {tab})",
                x.len(),
                cos.len(),
                sin.len()
            )));
        }
        if expected == 0 {
            return Ok(());
        }
        let mut d_x = self.htod(x)?;
        let d_cos = self.htod(cos)?;
        let d_sin = self.htod(sin)?;
        unsafe {
            self.launch_dit_rope(
                &mut d_x,
                &d_cos,
                &d_sin,
                num_heads as u32,
                seq as u32,
                head_dim as u32,
            )?;
        }
        self.dtoh_sync(&d_x, x)
    }

    // ── reshape: tokens → head-major ─────────────────────────────────────────

    /// Launch `tokens_to_heads_f32` (gather a token-major slice into head-major).
    ///
    /// # Safety
    /// `d_dst` holds ≥ `num_heads*seq*head_dim` f32; `d_src` covers
    /// `(seq-1)*src_stride + src_off + num_heads*head_dim`; on `self.stream`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn launch_dit_tokens_to_heads(
        &self,
        d_src: &CudaSlice<f32>,
        d_dst: &mut CudaSlice<f32>,
        seq: u32,
        num_heads: u32,
        head_dim: u32,
        src_stride: u32,
        src_off: u32,
    ) -> Result<(), CudaGraphError> {
        let cfg = LaunchConfig {
            grid_dim: (
                elementwise_grid(num_heads as usize * seq as usize * head_dim as usize),
                1,
                1,
            ),
            block_dim: (DIT_GLUE_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        self.stream
            .launch_builder(&self.modules.dit_tokens_to_heads)
            .arg(d_src)
            .arg(d_dst)
            .arg(&seq)
            .arg(&num_heads)
            .arg(&head_dim)
            .arg(&src_stride)
            .arg(&src_off)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("tokens_to_heads_f32 launch: {e}")))
    }

    /// Reshape a token-major `[seq, hidden]` slice (`hidden = num_heads*head_dim`,
    /// at column `src_off`, row stride `src_stride`) into a head-major contiguous
    /// `[num_heads, seq, head_dim]` buffer. With `src_stride = hidden, src_off = 0`
    /// this is exactly `oxibonsai-image::math::to_heads`.
    ///
    /// # Errors
    /// [`CudaGraphError::DriverError`] on a length mismatch or buffer/launch error.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_dit_tokens_to_heads(
        &self,
        src: &[f32],
        out: &mut [f32],
        seq: usize,
        num_heads: usize,
        head_dim: usize,
        src_stride: usize,
        src_off: usize,
    ) -> Result<(), CudaGraphError> {
        let out_len = num_heads
            .checked_mul(seq)
            .and_then(|x| x.checked_mul(head_dim))
            .ok_or_else(|| {
                CudaGraphError::DriverError("encode_dit_tokens_to_heads: out overflow".into())
            })?;
        let hidden = num_heads * head_dim;
        let need = if seq == 0 {
            0
        } else {
            (seq - 1) * src_stride + src_off + hidden
        };
        if out.len() != out_len || src.len() < need {
            return Err(CudaGraphError::DriverError(format!(
                "encode_dit_tokens_to_heads: bad lens (out {} != {out_len}, src {} < {need})",
                out.len(),
                src.len()
            )));
        }
        if out_len == 0 {
            return Ok(());
        }
        let d_src = self.htod(src)?;
        let mut d_out = self.alloc_zeros(out_len)?;
        unsafe {
            self.launch_dit_tokens_to_heads(
                &d_src,
                &mut d_out,
                seq as u32,
                num_heads as u32,
                head_dim as u32,
                src_stride as u32,
                src_off as u32,
            )?;
        }
        self.dtoh_sync(&d_out, out)
    }

    // ── reshape: strided per-row slice copy ──────────────────────────────────

    /// Launch `strided_row_copy_f32`: `dst[t,dst_off+j] = src[t,src_off+j]`.
    ///
    /// # Safety
    /// `d_dst`/`d_src` cover their `(rows-1)*stride + off + cols` ranges; on
    /// `self.stream`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn launch_dit_strided_row_copy(
        &self,
        d_dst: &mut CudaSlice<f32>,
        d_src: &CudaSlice<f32>,
        rows: u32,
        cols: u32,
        dst_stride: u32,
        dst_off: u32,
        src_stride: u32,
        src_off: u32,
    ) -> Result<(), CudaGraphError> {
        let cfg = LaunchConfig {
            grid_dim: (elementwise_grid(rows as usize * cols as usize), 1, 1),
            block_dim: (DIT_GLUE_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        self.stream
            .launch_builder(&self.modules.dit_strided_row_copy)
            .arg(d_dst)
            .arg(d_src)
            .arg(&rows)
            .arg(&cols)
            .arg(&dst_stride)
            .arg(&dst_off)
            .arg(&src_stride)
            .arg(&src_off)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("strided_row_copy_f32 launch: {e}")))
    }

    /// Per-row slice copy `dst[t, dst_off+j] = src[t, src_off+j]` (`j < cols`,
    /// `t < rows`). Extracts the mlp slab from a fused proj / builds the
    /// `[attn ‖ gated]` concat without per-row host copies.
    ///
    /// # Errors
    /// [`CudaGraphError::DriverError`] on a range mismatch or buffer/launch error.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_dit_strided_row_copy(
        &self,
        dst: &mut [f32],
        src: &[f32],
        rows: usize,
        cols: usize,
        dst_stride: usize,
        dst_off: usize,
        src_stride: usize,
        src_off: usize,
    ) -> Result<(), CudaGraphError> {
        if rows == 0 || cols == 0 {
            return Ok(());
        }
        let dst_need = (rows - 1) * dst_stride + dst_off + cols;
        let src_need = (rows - 1) * src_stride + src_off + cols;
        if dst.len() < dst_need || src.len() < src_need {
            return Err(CudaGraphError::DriverError(format!(
                "encode_dit_strided_row_copy: short (dst {} < {dst_need}, src {} < {src_need})",
                dst.len(),
                src.len()
            )));
        }
        let mut d_dst = self.htod(dst)?;
        let d_src = self.htod(src)?;
        unsafe {
            self.launch_dit_strided_row_copy(
                &mut d_dst,
                &d_src,
                rows as u32,
                cols as u32,
                dst_stride as u32,
                dst_off as u32,
                src_stride as u32,
                src_off as u32,
            )?;
        }
        self.dtoh_sync(&d_dst, dst)
    }

    // ── small shared host<->device helpers ──────────────────────────────────

    /// Upload a host slice to a fresh device buffer.
    pub(crate) fn htod(&self, src: &[f32]) -> Result<CudaSlice<f32>, CudaGraphError> {
        self.stream
            .clone_htod(src)
            .map_err(|e| CudaGraphError::DriverError(format!("dit glue htod: {e}")))
    }

    /// Allocate a zeroed device buffer.
    pub(crate) fn alloc_zeros(&self, len: usize) -> Result<CudaSlice<f32>, CudaGraphError> {
        self.stream
            .alloc_zeros::<f32>(len)
            .map_err(|e| CudaGraphError::DriverError(format!("dit glue alloc_zeros: {e}")))
    }

    /// Download a device buffer into `dst` and synchronize.
    pub(crate) fn dtoh_sync(
        &self,
        src: &CudaSlice<f32>,
        dst: &mut [f32],
    ) -> Result<(), CudaGraphError> {
        self.stream
            .memcpy_dtoh(src, dst)
            .map_err(|e| CudaGraphError::DriverError(format!("dit glue dtoh: {e}")))?;
        self.stream
            .synchronize()
            .map_err(|e| CudaGraphError::DriverError(format!("dit glue sync: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::CudaGraph;

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
            return 1.0;
        }
        (dot / (na.sqrt() * nb.sqrt())) as f32
    }
    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()))
    }
    /// Deterministic LCG fill in [-0.5, 0.5).
    fn lcg_fill(buf: &mut [f32], seed: u64) {
        let mut s = seed | 1;
        for v in buf.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5;
        }
    }

    macro_rules! graph_or_skip {
        () => {
            match CudaGraph::global() {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("no GPU, skip: {e}");
                    return;
                }
            }
        };
    }

    #[test]
    fn modulate_matches_cpu() {
        let _serial = super::super::types::gpu_parity_test_guard();
        let g = graph_or_skip!();
        let (rows, dim) = (40usize, 320usize);
        let mut x = vec![0f32; rows * dim];
        let mut shift = vec![0f32; dim];
        let mut scale = vec![0f32; dim];
        lcg_fill(&mut x, 1);
        lcg_fill(&mut shift, 2);
        lcg_fill(&mut scale, 3);
        let mut cpu = x.clone();
        for r in 0..rows {
            for i in 0..dim {
                let idx = r * dim + i;
                cpu[idx] = (1.0 + scale[i]) * cpu[idx] + shift[i];
            }
        }
        g.encode_dit_modulate(&mut x, &shift, &scale, rows, dim)
            .expect("encode_dit_modulate");
        assert!(cosine(&cpu, &x) >= 0.999 && max_abs(&cpu, &x) < 1e-3);
    }

    #[test]
    fn gated_residual_add_matches_cpu() {
        let _serial = super::super::types::gpu_parity_test_guard();
        let g = graph_or_skip!();
        let (rows, dim) = (48usize, 256usize);
        let mut h = vec![0f32; rows * dim];
        let mut delta = vec![0f32; rows * dim];
        let mut gate = vec![0f32; dim];
        lcg_fill(&mut h, 4);
        lcg_fill(&mut delta, 5);
        lcg_fill(&mut gate, 6);
        let mut cpu = h.clone();
        for r in 0..rows {
            for i in 0..dim {
                cpu[r * dim + i] += gate[i] * delta[r * dim + i];
            }
        }
        g.encode_dit_gated_residual_add(&mut h, &delta, &gate, rows, dim)
            .expect("encode_dit_gated_residual_add");
        assert!(cosine(&cpu, &h) >= 0.999 && max_abs(&cpu, &h) < 1e-3);
    }

    #[test]
    fn layer_norm_matches_cpu() {
        let _serial = super::super::types::gpu_parity_test_guard();
        let g = graph_or_skip!();
        let (rows, dim, eps) = (33usize, 512usize, 1e-6f32);
        let mut x = vec![0f32; rows * dim];
        lcg_fill(&mut x, 7);
        let mut cpu = x.clone();
        let inv = 1.0f32 / dim as f32;
        for r in 0..rows {
            let row = &mut cpu[r * dim..(r + 1) * dim];
            let mean: f32 = row.iter().sum::<f32>() * inv;
            let var: f32 = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() * inv;
            let inv_std = 1.0 / (var + eps).sqrt();
            for v in row.iter_mut() {
                *v = (*v - mean) * inv_std;
            }
        }
        g.encode_dit_layer_norm(&mut x, rows, dim, eps)
            .expect("encode_dit_layer_norm");
        assert!(
            cosine(&cpu, &x) >= 0.999 && max_abs(&cpu, &x) < 1e-3,
            "cos={} mae={}",
            cosine(&cpu, &x),
            max_abs(&cpu, &x)
        );
    }

    #[test]
    fn rms_norm_heads_matches_cpu() {
        let _serial = super::super::types::gpu_parity_test_guard();
        let g = graph_or_skip!();
        let (rows, head_dim, eps) = (50usize, 128usize, 1e-6f32);
        let mut x = vec![0f32; rows * head_dim];
        let mut w = vec![0f32; head_dim];
        lcg_fill(&mut x, 8);
        lcg_fill(&mut w, 9);
        let mut cpu = x.clone();
        let inv = 1.0f32 / head_dim as f32;
        for r in 0..rows {
            let row = &mut cpu[r * head_dim..(r + 1) * head_dim];
            let ms: f32 = row.iter().map(|&v| v * v).sum::<f32>() * inv;
            let inv_rms = 1.0 / (ms + eps).sqrt();
            for (i, v) in row.iter_mut().enumerate() {
                *v = w[i] * *v * inv_rms;
            }
        }
        g.encode_dit_rms_norm_heads(&mut x, &w, rows, head_dim, eps)
            .expect("encode_dit_rms_norm_heads");
        assert!(cosine(&cpu, &x) >= 0.999 && max_abs(&cpu, &x) < 1e-3);
    }

    #[test]
    fn swiglu_matches_cpu() {
        let _serial = super::super::types::gpu_parity_test_guard();
        let g = graph_or_skip!();
        let (rows, half) = (40usize, 384usize);
        let mut x = vec![0f32; rows * 2 * half];
        lcg_fill(&mut x, 10);
        let mut cpu = vec![0f32; rows * half];
        let full = 2 * half;
        for r in 0..rows {
            for i in 0..half {
                let gate = x[r * full + i];
                let up = x[r * full + half + i];
                cpu[r * half + i] = (gate / (1.0 + (-gate).exp())) * up;
            }
        }
        let mut out = vec![0f32; rows * half];
        g.encode_dit_swiglu(&x, &mut out, rows, half)
            .expect("encode_dit_swiglu");
        assert!(cosine(&cpu, &out) >= 0.999 && max_abs(&cpu, &out) < 1e-3);
    }

    #[test]
    fn rope_matches_cpu() {
        let _serial = super::super::types::gpu_parity_test_guard();
        let g = graph_or_skip!();
        let (num_heads, seq, head_dim) = (4usize, 50usize, 128usize);
        let half = head_dim / 2;
        let mut x = vec![0f32; num_heads * seq * head_dim];
        let mut cos = vec![0f32; seq * half];
        let mut sin = vec![0f32; seq * half];
        lcg_fill(&mut x, 11);
        // Real-ish rotary tables: cos/sin of per-position angles.
        for t in 0..seq {
            for i in 0..half {
                let theta = (t as f32) * (0.01 + 0.0003 * i as f32);
                cos[t * half + i] = theta.cos();
                sin[t * half + i] = theta.sin();
            }
        }
        let mut cpu = x.clone();
        for h in 0..num_heads {
            for t in 0..seq {
                let base = (h * seq + t) * head_dim;
                for i in 0..half {
                    let re = cpu[base + 2 * i];
                    let im = cpu[base + 2 * i + 1];
                    let c = cos[t * half + i];
                    let s = sin[t * half + i];
                    cpu[base + 2 * i] = re * c - im * s;
                    cpu[base + 2 * i + 1] = im * c + re * s;
                }
            }
        }
        g.encode_dit_rope(&mut x, &cos, &sin, num_heads, seq, head_dim)
            .expect("encode_dit_rope");
        assert!(cosine(&cpu, &x) >= 0.999 && max_abs(&cpu, &x) < 1e-3);
    }

    #[test]
    fn tokens_to_heads_matches_cpu() {
        let _serial = super::super::types::gpu_parity_test_guard();
        let g = graph_or_skip!();
        let (seq, num_heads, head_dim) = (50usize, 6usize, 64usize);
        let hidden = num_heads * head_dim;
        // Token-major src as a column slice (stride/off) of a wider fused proj.
        let (stride, off) = (hidden + 17, 9usize);
        let mut src = vec![0f32; seq * stride];
        lcg_fill(&mut src, 21);
        let mut cpu = vec![0f32; num_heads * seq * head_dim];
        for h in 0..num_heads {
            for t in 0..seq {
                for d in 0..head_dim {
                    cpu[(h * seq + t) * head_dim + d] = src[t * stride + off + h * head_dim + d];
                }
            }
        }
        let mut out = vec![0f32; num_heads * seq * head_dim];
        g.encode_dit_tokens_to_heads(&src, &mut out, seq, num_heads, head_dim, stride, off)
            .expect("encode_dit_tokens_to_heads");
        assert_eq!(cpu, out, "tokens_to_heads gather must be exact");
    }

    #[test]
    fn strided_row_copy_matches_cpu() {
        let _serial = super::super::types::gpu_parity_test_guard();
        let g = graph_or_skip!();
        let (rows, cols) = (40usize, 128usize);
        let (dst_stride, dst_off) = (cols + 20, 7usize);
        let (src_stride, src_off) = (cols + 13, 5usize);
        let mut dst = vec![0f32; rows * dst_stride];
        let mut src = vec![0f32; rows * src_stride];
        lcg_fill(&mut dst, 22);
        lcg_fill(&mut src, 23);
        let mut cpu = dst.clone();
        for t in 0..rows {
            for j in 0..cols {
                cpu[t * dst_stride + dst_off + j] = src[t * src_stride + src_off + j];
            }
        }
        g.encode_dit_strided_row_copy(
            &mut dst, &src, rows, cols, dst_stride, dst_off, src_stride, src_off,
        )
        .expect("encode_dit_strided_row_copy");
        assert_eq!(cpu, dst, "strided_row_copy must be exact");
    }
}
