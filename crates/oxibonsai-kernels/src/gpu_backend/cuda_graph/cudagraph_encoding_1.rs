//! # CudaGraph - encoding Methods
//!
//! This module contains method implementations for `CudaGraph`.
//!
//! Hardware status (RTX A4000, CUDA 12.0, 2026-10-07): `dtoh_exact` ran under
//! `encode_lm_head_gemv_tq2` (Step 3d, CUDA-P18). `encode_qkv_phase` serves
//! the per-block Q1 path (`try_cuda_qkv`), which no hardware test targeted —
//! not yet run on CUDA hardware by a dedicated test.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use cudarc::driver::CudaSlice;
use std::sync::Arc;

use super::types::{CudaGraphError, QkvBuffers};

use super::cudagraph_type::CudaGraph;

impl CudaGraph {
    /// Copy exactly `dst.len()` elements out of a **capacity-based** device
    /// buffer.
    ///
    /// Findings F2 and F8. The pooled QKV and LM-head buffers grow to the
    /// largest request seen and are never shrunk, while cudarc's
    /// `memcpy_dtoh` asserts `dst.len() >= src.len()` and `clone_dtoh` returns
    /// `src.len()` elements. Copying the *whole* device buffer therefore
    /// panicked outright on a later, smaller QKV request, and returned a logits
    /// vector padded with the previous model's tail from the LM head — both
    /// reachable by running two differently shaped models (multi-model or a
    /// draft model) through the process-global singleton. Sizing the copy by
    /// the current request fixes both, and is the only correct D2H shape for a
    /// pooled buffer.
    pub(crate) fn dtoh_exact(
        &self,
        src: &CudaSlice<f32>,
        dst: &mut [f32],
        what: &str,
    ) -> Result<(), CudaGraphError> {
        debug_assert!(
            src.len() >= dst.len(),
            "{what}: device buffer holds {} elements, {} requested",
            src.len(),
            dst.len()
        );
        let view = src.try_slice(0..dst.len()).ok_or_else(|| {
            CudaGraphError::DriverError(format!(
                "{what}: device buffer holds {} elements, {} requested",
                src.len(),
                dst.len()
            ))
        })?;
        self.stream
            .memcpy_dtoh(&view, dst)
            .map_err(|e| CudaGraphError::DriverError(format!("download {what}: {e}")))
    }

    /// Ensure QKV projection buffers are allocated for `(input_len, output_len)`.
    /// Re-allocates if the existing buffers are too small.
    fn acquire_qkv_buffers(
        &self,
        input_len: usize,
        output_len: usize,
    ) -> Result<std::sync::MutexGuard<'_, Option<QkvBuffers>>, CudaGraphError> {
        let mut guard = self
            .qkv_buffers
            .lock()
            .map_err(|_| CudaGraphError::LockPoisoned)?;
        let needs_alloc = match guard.as_ref() {
            Some(b) => !b.fits(input_len, output_len),
            None => true,
        };
        if needs_alloc {
            let alloc = |n: usize| -> Result<CudaSlice<f32>, CudaGraphError> {
                self.stream
                    .alloc_zeros::<f32>(n)
                    .map_err(|e| CudaGraphError::DriverError(format!("alloc_zeros qkv({n}): {e}")))
            };
            *guard = Some(QkvBuffers {
                d_input: alloc(input_len)?,
                d_output: alloc(output_len)?,
                input_capacity: input_len,
                output_capacity: output_len,
            });
        }
        Ok(guard)
    }
    /// Execute a QKV projection using pre-allocated device buffers.
    ///
    /// Eliminates per-call `cuMemAlloc`/`cuMemFree` that penalised the V1 path.
    /// Uses V8 (shared-mem input cache) when `k ≤ 48 KB threshold`, V7 otherwise.
    pub fn encode_qkv_phase(
        &self,
        input: &[f32],
        output: &mut [f32],
        weight_w: &Arc<CudaSlice<u8>>,
        n_rows: usize,
        k: usize,
    ) -> Result<(), CudaGraphError> {
        let mut qkv_guard = self.acquire_qkv_buffers(k, n_rows)?;
        let qkv = qkv_guard
            .as_mut()
            .ok_or_else(|| CudaGraphError::DriverError("qkv buffers not allocated".into()))?;
        self.stream
            .memcpy_htod(&input[..k], &mut qkv.d_input)
            .map_err(|e| CudaGraphError::DriverError(format!("upload qkv_input: {e}")))?;
        unsafe {
            match Self::v8_shared_bytes(k) {
                Some(smem) => self.launch_gemv_v8(
                    weight_w,
                    &qkv.d_input,
                    &mut qkv.d_output,
                    n_rows as u32,
                    k as u32,
                    smem,
                )?,
                None => self.launch_gemv_v7(
                    weight_w,
                    &qkv.d_input,
                    &mut qkv.d_output,
                    n_rows as u32,
                    k as u32,
                )?,
            }
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaGraphError::DriverError(format!("qkv stream sync: {e}")))?;
        // F2: `d_output` has *capacity* `output_capacity >= n_rows`; the copy
        // must be sized by this request, not by the buffer.
        self.dtoh_exact(&qkv.d_output, &mut output[..n_rows], "qkv_output")?;
        self.stream
            .synchronize()
            .map_err(|e| CudaGraphError::DriverError(format!("qkv D2H sync: {e}")))?;
        Ok(())
    }
}
