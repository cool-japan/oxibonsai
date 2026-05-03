//! # CudaGraph - encoding Methods
//!
//! This module contains method implementations for `CudaGraph`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use cudarc::driver::CudaSlice;

use super::types::{CudaGraphError, LmHeadBuffers};

use super::cudagraph_type::CudaGraph;

impl CudaGraph {
    fn acquire_lm_head_buffers(
        &self,
        hidden: usize,
        vocab: usize,
    ) -> Result<std::sync::MutexGuard<'_, Option<LmHeadBuffers>>, CudaGraphError> {
        let mut guard = self
            .lm_head_buffers
            .lock()
            .map_err(|_| CudaGraphError::LockPoisoned)?;
        let needs_alloc = match guard.as_ref() {
            Some(b) => !b.fits(hidden, vocab),
            None => true,
        };
        if needs_alloc {
            let alloc = |n: usize| -> Result<CudaSlice<f32>, CudaGraphError> {
                self.stream
                    .alloc_zeros::<f32>(n)
                    .map_err(|e| CudaGraphError::DriverError(format!("alloc lm_head({n}): {e}")))
            };
            *guard = Some(LmHeadBuffers {
                d_input: alloc(hidden)?,
                d_output: alloc(vocab)?,
                hidden_capacity: hidden,
                vocab_capacity: vocab,
            });
        }
        Ok(guard)
    }
    /// Run the ternary LM-head GEMV on GPU: `logits = lm_head_tq2_weight × normed`.
    ///
    /// Uploads `normed` (hidden_size floats), launches a single TQ2 GEMV
    /// (`gemv_tq2_g128_v1`), downloads logits.
    /// The weight is uploaded/cached via `get_or_upload_weight_tq2_soa` on first call.
    pub fn encode_lm_head_gemv_tq2(
        &self,
        normed: &[f32],
        handle_id: u64,
        weight_bytes: &[u8],
        vocab_size: usize,
        hidden_size: usize,
    ) -> Result<Vec<f32>, CudaGraphError> {
        let d_weight = self.get_or_upload_weight_tq2_soa(handle_id, weight_bytes)?;
        let mut buf_guard = self.acquire_lm_head_buffers(hidden_size, vocab_size)?;
        let bufs = buf_guard.as_mut().ok_or_else(|| {
            CudaGraphError::DriverError("lm_head_tq2 buffers not allocated".into())
        })?;
        self.stream
            .memcpy_htod(&normed[..hidden_size], &mut bufs.d_input)
            .map_err(|e| CudaGraphError::DriverError(format!("upload lm_head_tq2 input: {e}")))?;
        unsafe {
            self.launch_gemv_tq2_v1_pub(
                &d_weight,
                &bufs.d_input,
                &mut bufs.d_output,
                vocab_size as u32,
                hidden_size as u32,
            )?;
        }
        let result = self.stream.clone_dtoh(&bufs.d_output).map_err(|e| {
            CudaGraphError::DriverError(format!("download lm_head_tq2 logits: {e}"))
        })?;
        self.stream
            .synchronize()
            .map_err(|e| CudaGraphError::DriverError(format!("lm_head_tq2 D2H sync: {e}")))?;
        Ok(result)
    }

    /// Run the LM-head GEMV on GPU: `logits = lm_head_weight × normed`.
    ///
    /// Uploads `normed` (hidden_size floats) once, launches GEMV, downloads logits.
    /// The weight is cached on first call and reused across tokens.
    pub fn encode_lm_head_gemv(
        &self,
        normed: &[f32],
        handle_id: u64,
        weight_bytes: &[u8],
        vocab_size: usize,
        hidden_size: usize,
    ) -> Result<Vec<f32>, CudaGraphError> {
        let d_weight = self.get_or_upload_weight_soa(handle_id, weight_bytes)?;
        let mut buf_guard = self.acquire_lm_head_buffers(hidden_size, vocab_size)?;
        let bufs = buf_guard
            .as_mut()
            .ok_or_else(|| CudaGraphError::DriverError("lm_head buffers not allocated".into()))?;
        self.stream
            .memcpy_htod(&normed[..hidden_size], &mut bufs.d_input)
            .map_err(|e| CudaGraphError::DriverError(format!("upload lm_head input: {e}")))?;
        unsafe {
            self.launch_gemv_pub(
                &d_weight,
                &bufs.d_input,
                &mut bufs.d_output,
                vocab_size as u32,
                hidden_size as u32,
            )?;
        }
        let result = self
            .stream
            .clone_dtoh(&bufs.d_output)
            .map_err(|e| CudaGraphError::DriverError(format!("download logits: {e}")))?;
        self.stream
            .synchronize()
            .map_err(|e| CudaGraphError::DriverError(format!("lm_head D2H sync: {e}")))?;
        Ok(result)
    }
}
