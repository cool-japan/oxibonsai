//! Q1\_0\_g128 tensor types and 1-bit data access.
//!
//! Defines the [`BlockQ1_0G128`] structure matching the PrismML GGUF format
//! and the [`OneBitTensor`] wrapper for efficient tensor access.

use half::f16;

use crate::error::{BonsaiError, BonsaiResult};

/// Number of weights per Q1\_0\_g128 block.
pub const QK1_0_G128: usize = 128;

/// Size of a Q1\_0\_g128 block in bytes (2-byte FP16 scale + 16 bytes sign bits).
pub const BLOCK_SIZE_BYTES: usize = 18;

/// A single Q1\_0\_g128 quantized block.
///
/// Layout (18 bytes total):
/// - `d`: FP16 scale factor (2 bytes) — shared by all 128 weights
/// - `qs`: 128 sign bits packed into 16 bytes
///
/// Weight reconstruction: `w[i] = bit[i] ? +d : -d`
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct BlockQ1_0G128 {
    /// Scale factor (delta), FP16.
    pub d: f16,
    /// 128 sign bits packed into 16 bytes.
    pub qs: [u8; QK1_0_G128 / 8],
}

const _: () = assert!(std::mem::size_of::<BlockQ1_0G128>() == BLOCK_SIZE_BYTES);

impl BlockQ1_0G128 {
    /// Interpret a raw byte slice as a block reference (zero-copy).
    ///
    /// # Errors
    /// Returns [`BonsaiError::InvalidQuantBlockSize`] if `data` is shorter
    /// than [`BLOCK_SIZE_BYTES`], and [`BonsaiError::KQuantError`] if
    /// `data`'s address is not aligned to `align_of::<Self>()` (2, from the
    /// leading `d: f16`). The odd-address case used to be unchecked UB
    /// behind a `SAFETY` comment that claimed an alignment check which did
    /// not exist anywhere in this function (core-gguf-07 / sec-02); proved
    /// empirically that a deliberately odd-addressed slice was accepted and
    /// handed back as a live reference. Use [`Self::from_bytes_copied`] when
    /// `data` may legitimately be unaligned (e.g. a `general.alignment == 1`
    /// file) and a zero-copy view is not required.
    pub fn from_bytes(data: &[u8]) -> BonsaiResult<&Self> {
        if data.len() < BLOCK_SIZE_BYTES {
            return Err(BonsaiError::InvalidQuantBlockSize {
                format: "Q1_0_g128",
                expected: BLOCK_SIZE_BYTES,
                actual: data.len(),
            });
        }
        // Mirrors `BlockTQ2_0_g128::slice_from_bytes`
        // (`quant_ternary.rs`) — every sibling quant block type checks
        // `align_offset` before casting; this was the sole exception.
        let align = std::mem::align_of::<Self>();
        if data.as_ptr().align_offset(align) != 0 {
            return Err(BonsaiError::KQuantError {
                reason: format!("Q1_0_g128 from_bytes: pointer not {align}-byte aligned"),
            });
        }
        // SAFETY: BlockQ1_0G128 is repr(C) with known layout; length and
        // alignment are both checked above.
        let ptr = data.as_ptr() as *const BlockQ1_0G128;
        Ok(unsafe { &*ptr })
    }

    /// Interpret a raw byte slice as a slice of blocks (zero-copy).
    ///
    /// # Errors
    /// Returns [`BonsaiError::InvalidQuantBlockSize`] if `data.len()` is not
    /// a multiple of [`BLOCK_SIZE_BYTES`], and [`BonsaiError::KQuantError`]
    /// if `data`'s address is not aligned to `align_of::<Self>()`. See
    /// [`Self::from_bytes`] for why the alignment check exists, and
    /// [`Self::slice_from_bytes_copied`] for a fallback that never fails on
    /// alignment.
    pub fn slice_from_bytes(data: &[u8]) -> BonsaiResult<&[Self]> {
        if !data.len().is_multiple_of(BLOCK_SIZE_BYTES) {
            return Err(BonsaiError::InvalidQuantBlockSize {
                format: "Q1_0_g128",
                expected: BLOCK_SIZE_BYTES,
                actual: data.len(),
            });
        }
        // Mirrors `BlockTQ2_0_g128::slice_from_bytes` (`quant_ternary.rs`):
        // `align_of::<Self>()` is 2 (the leading `d: f16`), so an
        // odd-addressed slice would be instant UB once cast below. Proved
        // empirically before this fix: slicing at an odd byte offset gave
        // `ptr % 2 == 1` and this function still returned `Ok`.
        let align = std::mem::align_of::<Self>();
        if data.as_ptr().align_offset(align) != 0 {
            return Err(BonsaiError::KQuantError {
                reason: format!("Q1_0_g128 slice_from_bytes: pointer not {align}-byte aligned"),
            });
        }
        let count = data.len() / BLOCK_SIZE_BYTES;
        let ptr = data.as_ptr() as *const BlockQ1_0G128;
        // SAFETY: repr(C) layout with a compile-time size assert above;
        // length and alignment are both checked above; the returned slice's
        // lifetime is tied to the input slice's.
        Ok(unsafe { std::slice::from_raw_parts(ptr, count) })
    }

    /// Copy one block out of a (possibly misaligned) byte slice.
    ///
    /// Unlike [`Self::from_bytes`], this never fails due to pointer
    /// alignment: the two fields are read out of `data` byte-by-byte with no
    /// pointer cast at all, so there is no alignment requirement on `data`
    /// itself. Use this for a legitimately unaligned source (e.g. an
    /// arbitrary byte offset into an mmap'd file when `general.alignment`
    /// is not the default 32) when a zero-copy view is not required.
    ///
    /// # Errors
    /// Returns [`BonsaiError::InvalidQuantBlockSize`] if `data` is shorter
    /// than [`BLOCK_SIZE_BYTES`].
    pub fn from_bytes_copied(data: &[u8]) -> BonsaiResult<Self> {
        if data.len() < BLOCK_SIZE_BYTES {
            return Err(BonsaiError::InvalidQuantBlockSize {
                format: "Q1_0_g128",
                expected: BLOCK_SIZE_BYTES,
                actual: data.len(),
            });
        }
        // Layout is `d: f16` (2 bytes) then `qs: [u8; 16]` (repr(C)
        // preserves declaration order) — read both fields directly rather
        // than casting `data`'s (possibly misaligned) pointer.
        let d = f16::from_le_bytes([data[0], data[1]]);
        let mut qs = [0u8; QK1_0_G128 / 8];
        qs.copy_from_slice(&data[2..BLOCK_SIZE_BYTES]);
        Ok(Self { d, qs })
    }

    /// Copy a whole tensor's worth of blocks out of a (possibly misaligned)
    /// byte slice.
    ///
    /// The multi-block counterpart of [`Self::from_bytes_copied`]: every
    /// production caller of [`Self::slice_from_bytes`] hands it a whole
    /// tensor (many blocks), not a single 18-byte block, so this is the
    /// fallback that actually matters when a legitimately unaligned mmap
    /// would otherwise force an error on the zero-copy path.
    ///
    /// # Errors
    /// Returns [`BonsaiError::InvalidQuantBlockSize`] if `data.len()` is not
    /// a multiple of [`BLOCK_SIZE_BYTES`].
    pub fn slice_from_bytes_copied(data: &[u8]) -> BonsaiResult<Vec<Self>> {
        if !data.len().is_multiple_of(BLOCK_SIZE_BYTES) {
            return Err(BonsaiError::InvalidQuantBlockSize {
                format: "Q1_0_g128",
                expected: BLOCK_SIZE_BYTES,
                actual: data.len(),
            });
        }
        data.chunks_exact(BLOCK_SIZE_BYTES)
            .map(Self::from_bytes_copied)
            .collect()
    }

    /// Get the sign bit for weight at index `i` (0..127).
    /// Returns `true` for +d, `false` for -d.
    #[inline]
    pub fn sign_bit(&self, i: usize) -> bool {
        debug_assert!(i < QK1_0_G128);
        let byte_index = i / 8;
        let bit_offset = i % 8;
        (self.qs[byte_index] >> bit_offset) & 1 != 0
    }

    /// Get the reconstructed weight value at index `i`.
    #[inline]
    pub fn weight(&self, i: usize) -> f32 {
        let d = self.d.to_f32();
        if self.sign_bit(i) {
            d
        } else {
            -d
        }
    }
}

/// A 1-bit tensor backed by Q1\_0\_g128 blocks.
///
/// This wraps raw GGUF tensor data and provides typed access to blocks
/// without copying or dequantizing the entire tensor.
#[derive(Debug)]
pub struct OneBitTensor<'a> {
    /// Tensor name.
    pub name: String,
    /// Shape dimensions.
    pub shape: Vec<u64>,
    /// Raw block data.
    blocks: &'a [BlockQ1_0G128],
}

impl<'a> OneBitTensor<'a> {
    /// Create a 1-bit tensor from raw GGUF tensor data bytes.
    pub fn from_raw(name: String, shape: Vec<u64>, data: &'a [u8]) -> BonsaiResult<Self> {
        let blocks = BlockQ1_0G128::slice_from_bytes(data)?;
        Ok(Self {
            name,
            shape,
            blocks,
        })
    }

    /// Number of blocks in this tensor.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Total number of elements (weights) in this tensor.
    pub fn element_count(&self) -> usize {
        self.blocks.len() * QK1_0_G128
    }

    /// Get a reference to the block at the given index.
    pub fn block(&self, index: usize) -> &BlockQ1_0G128 {
        &self.blocks[index]
    }

    /// Get all blocks as a slice.
    pub fn blocks(&self) -> &[BlockQ1_0G128] {
        self.blocks
    }

    /// Dequantize all blocks to FP32 values.
    ///
    /// For the full tensor, this allocates and fills an output vector.
    /// For per-operation dequantization, use the kernel crate instead.
    pub fn dequantize_all(&self) -> Vec<f32> {
        let n = self.element_count();
        let mut output = vec![0.0f32; n];
        for (i, block) in self.blocks.iter().enumerate() {
            let d = block.d.to_f32();
            let base = i * QK1_0_G128;
            for j in 0..QK1_0_G128 {
                let byte_index = j / 8;
                let bit_offset = j % 8;
                let bit = (block.qs[byte_index] >> bit_offset) & 1;
                output[base + j] = if bit != 0 { d } else { -d };
            }
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_block(scale: f32, bits: [u8; 16]) -> BlockQ1_0G128 {
        BlockQ1_0G128 {
            d: f16::from_f32(scale),
            qs: bits,
        }
    }

    #[test]
    fn block_size_is_18_bytes() {
        assert_eq!(std::mem::size_of::<BlockQ1_0G128>(), 18);
    }

    #[test]
    fn all_ones_dequantize_to_positive() {
        let block = make_block(2.0, [0xFF; 16]);
        for i in 0..128 {
            assert!(block.sign_bit(i));
            assert!((block.weight(i) - 2.0).abs() < 0.01);
        }
    }

    #[test]
    fn all_zeros_dequantize_to_negative() {
        let block = make_block(3.0, [0x00; 16]);
        for i in 0..128 {
            assert!(!block.sign_bit(i));
            assert!((block.weight(i) + 3.0).abs() < 0.01);
        }
    }

    #[test]
    fn alternating_bits() {
        // 0xAA = 10101010 in binary: bits 1,3,5,7 set; 0,2,4,6 clear
        let block = make_block(1.0, [0xAA; 16]);
        for i in 0..128 {
            if i % 2 == 0 {
                assert!(!block.sign_bit(i), "bit {i} should be 0");
            } else {
                assert!(block.sign_bit(i), "bit {i} should be 1");
            }
        }
    }

    #[test]
    fn from_bytes_roundtrip() {
        let block = make_block(1.5, [0xFF; 16]);
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                &block as *const BlockQ1_0G128 as *const u8,
                BLOCK_SIZE_BYTES,
            )
        };
        let parsed = BlockQ1_0G128::from_bytes(bytes).expect("block parse should succeed");
        assert_eq!(parsed, &block);
    }

    #[test]
    fn one_bit_tensor_dequantize() {
        let block = make_block(2.0, [0xFF; 16]);
        // Borrow directly from `block`'s own (compiler-guaranteed-aligned)
        // stack address, rather than `.to_vec()`-copying into a fresh
        // `Vec<u8>` first: `align_of::<u8>() == 1`, so nothing guarantees a
        // `Vec<u8>` allocation is 2-byte aligned — real-world allocators
        // are generous about it, but Miri's is not (verified empirically:
        // this exact `.to_vec()` pattern gave an odd-addressed buffer under
        // Miri and tripped the alignment guard below on a perfectly
        // legitimate block).
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                &block as *const BlockQ1_0G128 as *const u8,
                BLOCK_SIZE_BYTES,
            )
        };
        let tensor = OneBitTensor::from_raw("test".to_string(), vec![128], bytes)
            .expect("tensor creation should succeed");
        assert_eq!(tensor.num_blocks(), 1);
        assert_eq!(tensor.element_count(), 128);

        let values = tensor.dequantize_all();
        for &v in &values {
            assert!((v - 2.0).abs() < 0.01);
        }
    }

    // ── core-gguf-07 / sec-02: alignment guard + copied fallback ──────────

    /// Copy a well-formed block's bytes into `dst[..BLOCK_SIZE_BYTES]`.
    fn write_block_bytes(dst: &mut [u8], scale: f32, bits: [u8; 16]) {
        let block = make_block(scale, bits);
        let src: &[u8] = unsafe {
            std::slice::from_raw_parts(
                &block as *const BlockQ1_0G128 as *const u8,
                BLOCK_SIZE_BYTES,
            )
        };
        dst[..BLOCK_SIZE_BYTES].copy_from_slice(src);
    }

    /// `n` well-formed blocks, back-to-back, preceded by 0 or 1 pad bytes —
    /// returns `(buffer, byte_range)` where `buffer[byte_range]` is exactly
    /// `n * BLOCK_SIZE_BYTES` long *and* guaranteed to have an odd base
    /// address (`align_of::<BlockQ1_0G128>() == 2`).
    ///
    /// The pad count is chosen *from the buffer's own actual base address*
    /// rather than assumed: a mainstream native allocator conventionally
    /// hands out an at-least-even address for any allocation (so "pad by
    /// 1" reliably lands on odd), but Miri's allocator does not honour that
    /// convention for a `Vec<u8>` (`align_of::<u8>() == 1` is all it owes),
    /// and was observed handing back an *odd* base address, which "pad by
    /// 1" would then land back on *even* — silently making the test
    /// vacuous. Computing the pad from the real base address is correct
    /// under both; returning the exact range (rather than an open
    /// `buf[pad..]`) keeps the slice length exactly `n * BLOCK_SIZE_BYTES`
    /// regardless of which pad value was chosen (the buffer itself always
    /// reserves 1 spare byte to make either choice fit).
    fn blocks_buffer_forcing_misalignment(n: usize) -> (Vec<u8>, std::ops::Range<usize>) {
        let mut buf = vec![0u8; 1 + n * BLOCK_SIZE_BYTES];
        let base_is_odd = !(buf.as_ptr() as usize).is_multiple_of(2);
        let pad = if base_is_odd { 0 } else { 1 };
        for i in 0..n {
            let start = pad + i * BLOCK_SIZE_BYTES;
            // All sign bits set (0xFF) so every `sign_bit(i)` reads `true`.
            write_block_bytes(&mut buf[start..start + BLOCK_SIZE_BYTES], 1.5, [0xFF; 16]);
        }
        (buf, pad..pad + n * BLOCK_SIZE_BYTES)
    }

    #[test]
    fn from_bytes_rejects_a_misaligned_pointer() {
        let (buf, range) = blocks_buffer_forcing_misalignment(1);
        let misaligned = &buf[range];
        assert_ne!(
            misaligned
                .as_ptr()
                .align_offset(std::mem::align_of::<BlockQ1_0G128>()),
            0,
            "test fixture must actually be misaligned to be meaningful"
        );
        match BlockQ1_0G128::from_bytes(misaligned) {
            Err(BonsaiError::KQuantError { reason }) => {
                assert!(reason.contains("aligned"), "reason: {reason}");
            }
            other => panic!("expected KQuantError for a misaligned pointer, got: {other:?}"),
        }
    }

    #[test]
    fn slice_from_bytes_rejects_a_misaligned_pointer() {
        let (buf, range) = blocks_buffer_forcing_misalignment(3);
        let misaligned = &buf[range];
        assert_ne!(
            misaligned
                .as_ptr()
                .align_offset(std::mem::align_of::<BlockQ1_0G128>()),
            0,
            "test fixture must actually be misaligned to be meaningful"
        );
        match BlockQ1_0G128::slice_from_bytes(misaligned) {
            Err(BonsaiError::KQuantError { .. }) => {}
            other => {
                panic!("expected KQuantError for a misaligned multi-block slice, got: {other:?}")
            }
        }
    }

    #[test]
    fn slice_from_bytes_still_accepts_a_properly_aligned_multi_block_buffer() {
        // Borrow directly from a real `[BlockQ1_0G128; 2]` array (which the
        // compiler guarantees is correctly aligned) rather than a fresh
        // `Vec<u8>`: `align_of::<u8>() == 1` does not guarantee a `Vec<u8>`
        // allocation happens to be 2-byte aligned (see
        // `blocks_buffer_forcing_misalignment`'s doc comment) — this is the
        // "happy path" counterpart, so it must not rely on the same
        // unguaranteed allocator behaviour it is meant to be independent of.
        let blocks = [make_block(0.5, [0x11; 16]), make_block(2.5, [0x22; 16])];
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(blocks.as_ptr() as *const u8, 2 * BLOCK_SIZE_BYTES)
        };
        let parsed = BlockQ1_0G128::slice_from_bytes(bytes).expect("aligned slice should succeed");
        assert_eq!(parsed.len(), 2);
        assert!((parsed[0].d.to_f32() - 0.5).abs() < 0.01);
        assert!((parsed[1].d.to_f32() - 2.5).abs() < 0.01);
    }

    #[test]
    fn from_bytes_copied_never_fails_on_a_misaligned_pointer() {
        let (buf, range) = blocks_buffer_forcing_misalignment(1);
        let misaligned = &buf[range];
        // The zero-copy path must reject this exact input...
        assert!(BlockQ1_0G128::from_bytes(misaligned).is_err());
        // ...while the copied fallback succeeds and reads the same values.
        let copied = BlockQ1_0G128::from_bytes_copied(misaligned)
            .expect("copied fallback must succeed on a misaligned source");
        assert!((copied.d.to_f32() - 1.5).abs() < 0.01);
        for i in 0..128 {
            assert!(copied.sign_bit(i));
        }
    }

    #[test]
    fn from_bytes_copied_rejects_a_too_short_buffer() {
        let short = [0u8; BLOCK_SIZE_BYTES - 1];
        match BlockQ1_0G128::from_bytes_copied(&short) {
            Err(BonsaiError::InvalidQuantBlockSize {
                format,
                expected,
                actual,
            }) => {
                assert_eq!(format, "Q1_0_g128");
                assert_eq!(expected, BLOCK_SIZE_BYTES);
                assert_eq!(actual, short.len());
            }
            other => panic!("expected InvalidQuantBlockSize, got: {other:?}"),
        }
    }

    #[test]
    fn slice_from_bytes_copied_never_fails_on_a_misaligned_pointer() {
        let n = 4;
        let (buf, range) = blocks_buffer_forcing_misalignment(n);
        let misaligned = &buf[range];
        assert_eq!(misaligned.len(), n * BLOCK_SIZE_BYTES);
        // The zero-copy path must reject this exact input...
        assert!(BlockQ1_0G128::slice_from_bytes(misaligned).is_err());
        // ...while the copied fallback succeeds, one block per source slot.
        let copied = BlockQ1_0G128::slice_from_bytes_copied(misaligned)
            .expect("copied fallback must succeed on a misaligned source");
        assert_eq!(copied.len(), n);
        for block in &copied {
            assert!((block.d.to_f32() - 1.5).abs() < 0.01);
        }
    }

    #[test]
    fn slice_from_bytes_copied_rejects_a_non_multiple_length() {
        let buf = vec![0u8; BLOCK_SIZE_BYTES + 1];
        match BlockQ1_0G128::slice_from_bytes_copied(&buf) {
            Err(BonsaiError::InvalidQuantBlockSize { format, .. }) => {
                assert_eq!(format, "Q1_0_g128");
            }
            other => panic!("expected InvalidQuantBlockSize, got: {other:?}"),
        }
    }

    #[test]
    fn legacy_and_new_block_size_errors_agree_on_the_expected_count() {
        // Both the zero-copy and copied paths must report the same
        // `expected`/`format` for the same malformed input (wave-1
        // addendum #1: migrated off the legacy `BonsaiError::InvalidBlockSize`).
        let short = [0u8; 5];
        for result in [
            BlockQ1_0G128::from_bytes(&short).map(|_| ()),
            BlockQ1_0G128::slice_from_bytes(&short).map(|_| ()),
            BlockQ1_0G128::from_bytes_copied(&short).map(|_| ()),
        ] {
            match result {
                Err(BonsaiError::InvalidQuantBlockSize {
                    format, expected, ..
                }) => {
                    assert_eq!(format, "Q1_0_g128");
                    assert_eq!(expected, BLOCK_SIZE_BYTES);
                }
                other => panic!("expected InvalidQuantBlockSize, got: {other:?}"),
            }
        }
    }
}
