//! Zero-copy `&[BlockX] → &[u8]` casts for every quant block type used by the
//! CUDA dispatch path.
//!
//! Each helper relies on the source `BlockX` being `#[repr(C)]` with a stable
//! `BLOCK_*_BYTES` element size.  The whole module is gated on `native-cuda` +
//! Linux/Windows via the parent `mod.rs` inner attribute, so no per-item
//! `#[cfg(...)]` is required here.

/// Convert a `BlockQ4_0` slice to raw bytes (zero-copy).
///
/// # Safety
/// `BlockQ4_0` is `#[repr(C)]` with `BLOCK_Q4_0_BYTES` (18) bytes per element.
pub(super) fn blocks_q4_0_as_bytes(blocks: &[oxibonsai_core::BlockQ4_0]) -> &[u8] {
    // SAFETY: BlockQ4_0 is #[repr(C)] with a well-defined 18-byte layout.
    unsafe {
        std::slice::from_raw_parts(
            blocks.as_ptr().cast::<u8>(),
            blocks.len() * oxibonsai_core::BLOCK_Q4_0_BYTES,
        )
    }
}

/// Convert a `BlockQ8_0` slice to raw bytes (zero-copy).
///
/// # Safety
/// `BlockQ8_0` is `#[repr(C)]` with `BLOCK_Q8_0_BYTES` (34) bytes per element.
pub(super) fn blocks_q8_0_as_bytes(blocks: &[oxibonsai_core::BlockQ8_0]) -> &[u8] {
    // SAFETY: BlockQ8_0 is #[repr(C)] with a well-defined 34-byte layout.
    unsafe {
        std::slice::from_raw_parts(
            blocks.as_ptr().cast::<u8>(),
            blocks.len() * oxibonsai_core::BLOCK_Q8_0_BYTES,
        )
    }
}

/// Convert a `BlockQ2K` slice to raw bytes (zero-copy).
///
/// # Safety
/// `BlockQ2K` is `#[repr(C)]` with `BLOCK_Q2_K_BYTES` (84) bytes per element.
pub(super) fn blocks_q2k_as_bytes(blocks: &[oxibonsai_core::BlockQ2K]) -> &[u8] {
    // SAFETY: BlockQ2K is #[repr(C)] with a well-defined 84-byte layout.
    unsafe {
        std::slice::from_raw_parts(
            blocks.as_ptr().cast::<u8>(),
            blocks.len() * oxibonsai_core::BLOCK_Q2_K_BYTES,
        )
    }
}

/// Convert a `BlockQ3K` slice to raw bytes (zero-copy).
///
/// # Safety
/// `BlockQ3K` is `#[repr(C)]` with `BLOCK_Q3K_BYTES` (110) bytes per element.
pub(super) fn blocks_q3k_as_bytes(blocks: &[oxibonsai_core::BlockQ3K]) -> &[u8] {
    // SAFETY: BlockQ3K is #[repr(C)] with a well-defined 110-byte layout.
    unsafe {
        std::slice::from_raw_parts(
            blocks.as_ptr().cast::<u8>(),
            blocks.len() * oxibonsai_core::BLOCK_Q3K_BYTES,
        )
    }
}

/// Convert a `BlockQ4K` slice to raw bytes (zero-copy).
///
/// # Safety
/// `BlockQ4K` is `#[repr(C)]` with `BLOCK_Q4_K_BYTES` (144) bytes per element.
pub(super) fn blocks_q4k_as_bytes(blocks: &[oxibonsai_core::BlockQ4K]) -> &[u8] {
    // SAFETY: BlockQ4K is #[repr(C)] with a well-defined 144-byte layout.
    unsafe {
        std::slice::from_raw_parts(
            blocks.as_ptr().cast::<u8>(),
            blocks.len() * oxibonsai_core::BLOCK_Q4_K_BYTES,
        )
    }
}

/// Convert a `BlockQ5K` slice to raw bytes (zero-copy).
///
/// # Safety
/// `BlockQ5K` is `#[repr(C)]` with `BLOCK_Q5K_BYTES` (176) bytes per element.
pub(super) fn blocks_q5k_as_bytes(blocks: &[oxibonsai_core::BlockQ5K]) -> &[u8] {
    // SAFETY: BlockQ5K is #[repr(C)] with a well-defined 176-byte layout.
    unsafe {
        std::slice::from_raw_parts(
            blocks.as_ptr().cast::<u8>(),
            blocks.len() * oxibonsai_core::BLOCK_Q5K_BYTES,
        )
    }
}

/// Convert a `BlockQ6K` slice to raw bytes (zero-copy).
///
/// # Safety
/// `BlockQ6K` is `#[repr(C)]` with `BLOCK_Q6K_BYTES` (210) bytes per element.
pub(super) fn blocks_q6k_as_bytes(blocks: &[oxibonsai_core::BlockQ6K]) -> &[u8] {
    // SAFETY: BlockQ6K is #[repr(C)] with a well-defined 210-byte layout.
    unsafe {
        std::slice::from_raw_parts(
            blocks.as_ptr().cast::<u8>(),
            blocks.len() * oxibonsai_core::BLOCK_Q6K_BYTES,
        )
    }
}

/// Convert a `BlockQ8K` slice to raw bytes (zero-copy).
///
/// # Safety
/// `BlockQ8K` is `#[repr(C)]` with `BLOCK_Q8K_BYTES` (292) bytes per element.
pub(super) fn blocks_q8k_as_bytes(blocks: &[oxibonsai_core::BlockQ8K]) -> &[u8] {
    // SAFETY: BlockQ8K is #[repr(C)] with a well-defined 292-byte layout.
    unsafe {
        std::slice::from_raw_parts(
            blocks.as_ptr().cast::<u8>(),
            blocks.len() * oxibonsai_core::BLOCK_Q8K_BYTES,
        )
    }
}
