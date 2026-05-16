//! Intermediate buffer set for the FFN pipeline, plus low-level allocation,
//! upload/download, and dispatch helpers shared across the directory module.

use metal::{Buffer, Device, MTLResourceOptions};
use std::ffi::c_void;

use super::error::MetalGraphError;

// ═══════════════════════════════════════════════════════════════════════════
// Pre-allocated GPU buffers for the FFN pipeline
// ═══════════════════════════════════════════════════════════════════════════

/// Lazily allocated intermediate buffers used by `encode_ffn_phase`.
pub(super) struct MetalBuffers {
    pub(super) hidden_buf: Buffer,
    pub(super) attn_out_buf: Buffer,
    pub(super) norm_weight_buf: Buffer,
    pub(super) proj_buf: Buffer,
    pub(super) normed_buf: Buffer,
    pub(super) swiglu_buf: Buffer,
    pub(super) down_buf: Buffer,
    /// Hidden dimension these buffers were allocated for.
    pub(super) hidden_size: usize,
    /// Intermediate dimension (gate/up half size).
    pub(super) intermediate_size: usize,
}

impl MetalBuffers {
    /// Allocate all intermediate buffers for the given dimensions.
    pub(super) fn allocate(
        device: &Device,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<Self, MetalGraphError> {
        let h_bytes = (hidden_size * std::mem::size_of::<f32>()) as u64;
        let inter_bytes = (intermediate_size * std::mem::size_of::<f32>()) as u64;
        let shared = MTLResourceOptions::StorageModeShared;
        let private = MTLResourceOptions::StorageModePrivate;

        Ok(Self {
            hidden_buf: alloc_buf(device, h_bytes, shared)?, // CPU upload/download
            attn_out_buf: alloc_buf(device, h_bytes, shared)?, // CPU upload
            norm_weight_buf: alloc_buf(device, h_bytes, shared)?, // CPU upload
            proj_buf: alloc_buf(device, h_bytes, private)?,  // GPU-only intermediate
            normed_buf: alloc_buf(device, h_bytes, private)?, // GPU-only intermediate
            swiglu_buf: alloc_buf(device, inter_bytes, private)?, // GPU-only intermediate

            down_buf: alloc_buf(device, h_bytes, private)?, // GPU-only intermediate
            hidden_size,
            intermediate_size,
        })
    }

    /// Check whether existing buffers match the requested dimensions.
    pub(super) fn matches(&self, hidden_size: usize, intermediate_size: usize) -> bool {
        self.hidden_size == hidden_size && self.intermediate_size == intermediate_size
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Pre-allocated GPU buffers for the attention pipeline
// ═══════════════════════════════════════════════════════════════════════════

/// Helper: allocate a Metal buffer, converting a null pointer into an error.
pub(crate) fn alloc_buf(
    device: &Device,
    byte_len: u64,
    opts: MTLResourceOptions,
) -> Result<Buffer, MetalGraphError> {
    if byte_len == 0 {
        return Err(MetalGraphError::BufferCreationFailed);
    }
    let buf = device.new_buffer(byte_len, opts);
    // StorageModePrivate buffers have contents() == null by design
    if opts.contains(MTLResourceOptions::StorageModePrivate) {
        // For private buffers, just check length as a sanity proxy
        if buf.length() < byte_len {
            return Err(MetalGraphError::BufferCreationFailed);
        }
    } else if buf.contents().is_null() {
        return Err(MetalGraphError::BufferCreationFailed);
    }
    Ok(buf)
}

// ═══════════════════════════════════════════════════════════════════════════
// Upload / download helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Copy a host `f32` slice into a shared Metal buffer.
///
/// # Safety
///
/// The buffer must have been allocated with `StorageModeShared` and must be
/// large enough to hold `data.len()` floats.
pub(crate) unsafe fn upload_f32(buf: &Buffer, data: &[f32]) {
    std::ptr::copy_nonoverlapping(data.as_ptr(), buf.contents() as *mut f32, data.len());
}

/// Copy from a shared Metal buffer into a host `f32` slice.
///
/// # Safety
///
/// The buffer must have been allocated with `StorageModeShared` and must
/// contain at least `out.len()` floats of valid data.
pub(crate) unsafe fn download_f32(buf: &Buffer, out: &mut [f32]) {
    std::ptr::copy_nonoverlapping(buf.contents() as *const f32, out.as_mut_ptr(), out.len());
}

/// Upload raw bytes (weight data) into a GPU-accessible Metal buffer.
///
/// Uses `StorageModeShared` so the CPU can write directly and the GPU
/// can read without an explicit blit copy.
pub(super) fn upload_bytes(device: &Device, data: &[u8]) -> Result<Buffer, MetalGraphError> {
    if data.is_empty() {
        return Err(MetalGraphError::BufferCreationFailed);
    }
    let opts = MTLResourceOptions::StorageModeShared;
    let buf = device.new_buffer(data.len() as u64, opts);
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), buf.contents() as *mut u8, data.len());
    }
    Ok(buf)
}

// ═══════════════════════════════════════════════════════════════════════════
// Dispatch helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Compute threadgroup count: `ceil(n / divisor)`, guaranteed >= 1.
#[inline]
pub(crate) fn div_ceil(n: usize, divisor: usize) -> usize {
    n.div_ceil(divisor)
}

/// Convenience: `set_bytes` for a single scalar value at a given buffer index.
///
/// # Safety
///
/// The encoder must be in a valid state and `index` must not collide with
/// any buffer binding.
pub(crate) unsafe fn set_scalar<T: Copy>(
    encoder: &metal::ComputeCommandEncoderRef,
    index: u64,
    value: &T,
) {
    encoder.set_bytes(
        index,
        std::mem::size_of::<T>() as u64,
        value as *const T as *const c_void,
    );
}
