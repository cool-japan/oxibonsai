//! # CudaGraph - encoding Methods
//!
//! This module contains method implementations for `CudaGraph`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use std::sync::Arc;

use super::types::{CudaGraphError, TernaryGemvBuffers};

use super::cudagraph_type::CudaGraph;

impl CudaGraph {
    /// Launch `gemv_tq2_g128_v1` on the default stream.
    ///
    /// # Safety
    /// Caller must ensure all slices are valid device pointers on `self.stream`.
    unsafe fn launch_gemv_tq2_v1(
        &self,
        d_weight: &CudaSlice<u8>,
        d_input: &CudaSlice<f32>,
        d_output: &mut CudaSlice<f32>,
        n_rows: u32,
        k: u32,
    ) -> Result<(), CudaGraphError> {
        let grid_x = n_rows.div_ceil(8);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        self.stream
            .launch_builder(&self.modules.gemv_tq2_g128_v1)
            .arg(d_weight)
            .arg(d_input)
            .arg(d_output)
            .arg(&n_rows)
            .arg(&k)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("gemv_tq2_v1 launch: {e}")))
    }
    /// Public wrapper — launch `gemv_tq2_g128_v1` directly from cached device slices.
    ///
    /// Used by the full-forward ternary path (`encode_layer_into_ternary`) where the
    /// weight is already on device and the input/output slices live in the shared
    /// `CudaFullLayerBuffers` — no H2D/D2H or pool allocation is needed.
    ///
    /// # Safety
    /// All slices must be valid device pointers allocated on `self.stream`.
    pub unsafe fn launch_gemv_tq2_v1_pub(
        &self,
        d_weight: &Arc<CudaSlice<u8>>,
        d_input: &CudaSlice<f32>,
        d_output: &mut CudaSlice<f32>,
        n_rows: u32,
        k: u32,
    ) -> Result<(), CudaGraphError> {
        self.launch_gemv_tq2_v1(d_weight, d_input, d_output, n_rows, k)
    }

    /// Execute a TQ2 (ternary) GEMV using a pre-cached SoA weight handle.
    ///
    /// Uses a process-wide reusable input/output buffer pool that grows to fit
    /// the largest GEMV seen so far — eliminates the per-call cuMemAlloc/Free
    /// round-trip that otherwise dominates short-kernel dispatch overhead.
    pub fn encode_gemv_tq2_cached(
        &self,
        weight_id: u64,
        input: &[f32],
        n_rows: usize,
        k: usize,
    ) -> Result<Vec<f32>, CudaGraphError> {
        let d_weight = {
            let cache = self
                .weight_cache
                .lock()
                .map_err(|_| CudaGraphError::LockPoisoned)?;
            cache
                .get(&weight_id)
                .map(Arc::clone)
                .ok_or(CudaGraphError::WeightNotFound(weight_id))?
        };
        let mut buf_guard = self
            .tq2_gemv_buffers
            .lock()
            .map_err(|_| CudaGraphError::LockPoisoned)?;
        let needs_alloc = match buf_guard.as_ref() {
            Some(b) => !b.fits(k, n_rows),
            None => true,
        };
        if needs_alloc {
            let in_cap = match buf_guard.as_ref() {
                Some(b) => b.input_capacity.max(k),
                None => k,
            };
            let out_cap = match buf_guard.as_ref() {
                Some(b) => b.output_capacity.max(n_rows),
                None => n_rows,
            };
            let d_input = self.stream.alloc_zeros::<f32>(in_cap).map_err(|e| {
                CudaGraphError::DriverError(format!("alloc_zeros tq2 input pool: {e}"))
            })?;
            let d_output = self.stream.alloc_zeros::<f32>(out_cap).map_err(|e| {
                CudaGraphError::DriverError(format!("alloc_zeros tq2 output pool: {e}"))
            })?;
            *buf_guard = Some(TernaryGemvBuffers {
                d_input,
                d_output,
                input_capacity: in_cap,
                output_capacity: out_cap,
            });
        }
        let bufs = buf_guard
            .as_mut()
            .ok_or_else(|| CudaGraphError::DriverError("tq2 gemv buffers missing".into()))?;
        {
            let mut d_in_view = bufs.d_input.slice_mut(0..k);
            self.stream
                .memcpy_htod(&input[..k], &mut d_in_view)
                .map_err(|e| CudaGraphError::DriverError(format!("memcpy_htod tq2 input: {e}")))?;
        }
        unsafe {
            self.launch_gemv_tq2_v1(
                &d_weight,
                &bufs.d_input,
                &mut bufs.d_output,
                n_rows as u32,
                k as u32,
            )?;
        }
        let mut host = vec![0.0f32; n_rows];
        {
            let d_out_view = bufs.d_output.slice(0..n_rows);
            self.stream
                .memcpy_dtoh(&d_out_view, &mut host[..n_rows])
                .map_err(|e| CudaGraphError::DriverError(format!("memcpy_dtoh tq2 output: {e}")))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaGraphError::DriverError(format!("stream sync tq2: {e}")))?;
        Ok(host)
    }
}
