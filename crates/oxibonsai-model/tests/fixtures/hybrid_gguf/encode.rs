//! GGUF byte encoding of a planned tensor: plain `F32`, `BF16`, the four
//! block formats the library quantizes, and the fixture's own `Q1_0_g128`
//! packer.

use half::f16;

use oxibonsai_core::bf16::f32_to_bf16;
use oxibonsai_core::error::{BonsaiError, BonsaiResult};
use oxibonsai_core::gguf::writer::TensorType;
use oxibonsai_core::{BlockPQ2_0, BlockPTQ1_0, BlockQ2_0G64, BlockTQ2_0_g128};

use super::plan::TensorKind;

// ═════════════════════════════════════════════════════════════════════════
// 9. GGUF byte encoding
// ═════════════════════════════════════════════════════════════════════════

/// Reinterpret a `#[repr(C)]` `Copy` block slice as raw little-endian bytes
/// (every block type here packs `u8`/`f16` fields with no implicit padding,
/// verified by each type's own `size_of` const-assert upstream).
///
/// # Safety
/// `T` must be `#[repr(C)]`, `Copy`, and free of padding bytes — true for
/// every block type passed to this function (`BlockPQ2_0`, `BlockPTQ1_0`,
/// `BlockQ2_0G64`, `BlockTQ2_0_g128`), each of which carries its own
/// `size_of::<Self>() == BLOCK_*_BYTES` const-assert in `oxibonsai-core`.
fn blocks_to_bytes<T: Copy>(blocks: &[T]) -> Vec<u8> {
    let byte_len = std::mem::size_of_val(blocks);
    // SAFETY: see the function doc; `blocks` outlives the `slice::from_raw_parts`
    // call and the resulting slice is copied into an owned `Vec` before
    // `blocks` could be dropped or mutated.
    unsafe { std::slice::from_raw_parts(blocks.as_ptr() as *const u8, byte_len) }.to_vec()
}

/// Hand-pack a `Q1_0_g128` tensor: `d: f16` first, then 128 sign bits per
/// block (`bit == 1 -> +d`). No library `quantize()` exists for this
/// 1-bit-only format (design §1.4 covers `PQ2_0`/`PTQ1_0`/`Q2_0_g64` only),
/// so this fixture supplies its own — every value here is exactly `+1.0`
/// or `-1.0`, so `d = 1.0` (exact in `f16`) makes the encoding lossless.
fn encode_q1_0_g128(values: &[f32]) -> BonsaiResult<Vec<u8>> {
    if !values.len().is_multiple_of(128) {
        return Err(BonsaiError::KQuantError {
            reason: format!(
                "Q1_0_g128 fixture encode: length {} is not a multiple of 128",
                values.len()
            ),
        });
    }
    let mut out = Vec::with_capacity(values.len() / 128 * 18);
    for chunk in values.chunks_exact(128) {
        out.extend_from_slice(&f16::from_f32(1.0).to_le_bytes());
        let mut qs = [0u8; 16];
        for (i, &v) in chunk.iter().enumerate() {
            if v > 0.0 {
                qs[i / 8] |= 1 << (i % 8);
            }
        }
        out.extend_from_slice(&qs);
    }
    Ok(out)
}

fn encode_f32(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn encode_bf16(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for &v in values {
        out.extend_from_slice(&f32_to_bf16(v).to_le_bytes());
    }
    out
}

fn quantize_matrix(quant: TensorType, values: &[f32]) -> BonsaiResult<Vec<u8>> {
    match quant {
        TensorType::F32 => Ok(encode_f32(values)),
        TensorType::PQ2_0 => BlockPQ2_0::quantize(values).map(|b| blocks_to_bytes(&b)),
        TensorType::PTQ1_0 => BlockPTQ1_0::quantize(values).map(|b| blocks_to_bytes(&b)),
        TensorType::Q2_0G64 => BlockQ2_0G64::quantize(values).map(|b| blocks_to_bytes(&b)),
        TensorType::TQ2_0_g128 => BlockTQ2_0_g128::quantize(values).map(|b| blocks_to_bytes(&b)),
        TensorType::Q1_0G128 => encode_q1_0_g128(values),
        other => Err(BonsaiError::KQuantError {
            reason: format!("hybrid fixture: {other:?} is not a supported quant choice"),
        }),
    }
}

pub(super) fn encode_tensor(
    kind: TensorKind,
    quant: TensorType,
    values: &[f32],
) -> BonsaiResult<(TensorType, Vec<u8>)> {
    match kind {
        TensorKind::PlainF32 => Ok((TensorType::F32, encode_f32(values))),
        TensorKind::Bf16 => Ok((TensorType::BF16, encode_bf16(values))),
        TensorKind::Quant => Ok((quant, quantize_matrix(quant, values)?)),
    }
}
