//! # CudaGraph - accessors Methods
//!
//! This module contains method implementations for `CudaGraph`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use cudarc::driver::CudaSlice;
use std::sync::Arc;

use super::types::CudaGraphError;

use super::cudagraph_type::CudaGraph;

impl CudaGraph {
    /// Like [`Self::get_or_upload_weight_soa`] but accepts a lazy byte producer.
    ///
    /// The closure is only called on the first use of `handle_id`.  Useful when
    /// the caller needs to concatenate gate+up bytes without computing them on
    /// every token.
    pub fn get_or_upload_weight_soa_lazy<F>(
        &self,
        handle_id: u64,
        make_bytes: F,
    ) -> Result<Arc<CudaSlice<u8>>, CudaGraphError>
    where
        F: FnOnce() -> Vec<u8>,
    {
        {
            let cache = self
                .weight_cache
                .lock()
                .map_err(|_| CudaGraphError::LockPoisoned)?;
            if let Some(existing) = cache.get(&handle_id) {
                return Ok(Arc::clone(existing));
            }
        }
        let aos_bytes = make_bytes();
        self.get_or_upload_weight_soa(handle_id, &aos_bytes)
    }
}
