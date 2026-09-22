//! Q1 / TQ2 / PQ2 weight block reformatters (AoS → SoA) and ternary code validation.
//!
//! The Metal MSL kernels expect a Structure-of-Arrays layout where all scales
//! come first and all packed quant data follows, in order to maximize
//! coalesced reads.  These helpers translate the on-disk Array-of-Structures
//! layout into the SoA layout consumed by the GPU.
//!
//! # Two mirror-image 34-byte block layouts
//!
//! `TQ2_0_g128` (our legacy ggml type 42) and PrismML's `PQ2_0` (ggml type 142)
//! both pack 128 weights into 34 bytes as `32 B` of LSB-first 2-bit codes plus
//! one `f16` scale — but the two orders are **mirror images**:
//!
//! | Format       | bytes `0..2` | bytes `2..34`                 | reserved code `0b11`            |
//! |--------------|--------------|-------------------------------|---------------------------------|
//! | `TQ2_0_g128` | `qs[0..2]`   | `qs[2..32]` + `d` at `32..34` | decodes to `0.0` (never emitted) |
//! | `PQ2_0`      | `d`          | `qs[0..32]`                   | decodes to `+2.0` (legal)        |
//!
//! `aos_bytes.len() % 34 == 0` holds for both, so feeding a `PQ2_0` tensor to
//! [`reformat_tq2_aos_to_soa`] does **not** fail — it silently reads `qs[0..2]`
//! as the scale and `d ++ qs[0..30]` as quant data. Each format therefore gets
//! its own reformatter, and a tensor *declared* ternary is additionally screened
//! for the reserved `0b11` code by [`validate_tq2_ternary_codes`], which is the
//! cheap signal that a `PQ2_0` tensor was routed into the ternary path.

use std::fmt;

// ═══════════════════════════════════════════════════════════════════════════
// Shared 2-bit block geometry (TQ2_0_g128 and PQ2_0)
// ═══════════════════════════════════════════════════════════════════════════

/// Bytes per 2-bit block: 32 B of packed codes + one `f16` scale.
pub(super) const Q2_BLOCK_BYTES: usize = 34;
/// Bytes of `f16` scale per 2-bit block.
pub(super) const Q2_SCALE_BYTES: usize = 2;
/// Bytes of packed 2-bit codes per block (128 weights, 4 per byte, LSB-first).
pub(super) const Q2_QS_BYTES: usize = 32;
/// Weights per 2-bit block.
pub(super) const Q2_BLOCK_WEIGHTS: usize = 128;

// ═══════════════════════════════════════════════════════════════════════════
// Q1 AoS → SoA reformatter
// ═══════════════════════════════════════════════════════════════════════════

/// Reformat Q1_0_g128 weight bytes from AoS to SoA layout.
///
/// AoS (input):  [scale₀|data₀][scale₁|data₁]...[scaleₙ|dataₙ]
///   Each block: 2 bytes FP16 scale + 16 bytes sign data = 18 bytes
///
/// SoA (output): [scale₀|scale₁|...|scaleₙ][data₀|data₁|...|dataₙ]
///   Scales section: N × 2 bytes (sequential, perfectly coalesced)
///   Data section:   N × 16 bytes (16-byte aligned, uint4 loads)
///
/// Total size is unchanged: N × 18 bytes.
/// Returns `None` if the input length is not a multiple of 18.
pub(super) fn reformat_q1_aos_to_soa(aos_bytes: &[u8]) -> Option<Vec<u8>> {
    const BLOCK_SIZE: usize = 18;
    const SCALE_SIZE: usize = 2;
    const DATA_SIZE: usize = 16;

    if aos_bytes.is_empty() || !aos_bytes.len().is_multiple_of(BLOCK_SIZE) {
        return None;
    }

    let n_blocks = aos_bytes.len() / BLOCK_SIZE;
    let mut soa = vec![0u8; n_blocks * BLOCK_SIZE];

    let (scales_section, data_section) = soa.split_at_mut(n_blocks * SCALE_SIZE);

    for i in 0..n_blocks {
        let block_start = i * BLOCK_SIZE;
        // Copy scale (2 bytes) to scales section
        scales_section[i * SCALE_SIZE..i * SCALE_SIZE + SCALE_SIZE]
            .copy_from_slice(&aos_bytes[block_start..block_start + SCALE_SIZE]);
        // Copy data (16 bytes) to data section
        data_section[i * DATA_SIZE..i * DATA_SIZE + DATA_SIZE]
            .copy_from_slice(&aos_bytes[block_start + SCALE_SIZE..block_start + BLOCK_SIZE]);
    }

    Some(soa)
}

// ═══════════════════════════════════════════════════════════════════════════
// TQ2_0_g128 (qs-first) AoS → SoA reformatter
// ═══════════════════════════════════════════════════════════════════════════

/// Reformat TQ2_0_g128 AoS → SoA for the Metal ternary GEMV kernel.
///
/// AoS (input) block layout: `{ qs: [u8; 32], d: f16 }` = 34 bytes (**qs first**)
/// SoA (output) layout: `[all d: N × 2 bytes FP16 LE][all qs: N × 32 bytes]`
/// Note: the ternary MSL kernel consumes this layout directly — scales first,
/// then qs data, matching the convention in `scirs2_backend::upload_weights_ternary`.
///
/// Total size is unchanged: N × 34 bytes.
/// Returns `None` if the input length is not a multiple of 34.
///
/// See [`reformat_pq2_aos_to_soa`] for the `d`-first (`PQ2_0`) sibling — the two
/// inputs are indistinguishable by length, so the caller must pick the right one.
pub(super) fn reformat_tq2_aos_to_soa(aos_bytes: &[u8]) -> Option<Vec<u8>> {
    if aos_bytes.is_empty() || !aos_bytes.len().is_multiple_of(Q2_BLOCK_BYTES) {
        return None;
    }

    let n_blocks = aos_bytes.len() / Q2_BLOCK_BYTES;
    let mut soa = vec![0u8; n_blocks * Q2_BLOCK_BYTES];

    let (scales_section, data_section) = soa.split_at_mut(n_blocks * Q2_SCALE_BYTES);

    for i in 0..n_blocks {
        let block_start = i * Q2_BLOCK_BYTES;
        // TQ2 AoS: first 32 bytes are qs, last 2 are scale. SoA: scales first, then qs.
        data_section[i * Q2_QS_BYTES..i * Q2_QS_BYTES + Q2_QS_BYTES]
            .copy_from_slice(&aos_bytes[block_start..block_start + Q2_QS_BYTES]);
        scales_section[i * Q2_SCALE_BYTES..i * Q2_SCALE_BYTES + Q2_SCALE_BYTES]
            .copy_from_slice(&aos_bytes[block_start + Q2_QS_BYTES..block_start + Q2_BLOCK_BYTES]);
    }

    Some(soa)
}

// ═══════════════════════════════════════════════════════════════════════════
// PQ2_0 (d-first) AoS → SoA reformatter
// ═══════════════════════════════════════════════════════════════════════════

/// Reformat `PQ2_0` (PrismML ggml type **142**) AoS → SoA for the Metal
/// 2-bit GEMV/GEMM kernels.
///
/// AoS (input) block layout — the mirror image of `TQ2_0_g128`:
/// `{ d: f16, qs: [u8; 32] }` = 34 bytes (**`d` first**, bytes `0..2`;
/// `qs` at bytes `2..34`), matching `block_pq2_0` in the PrismML llama.cpp fork
/// (`ggml-common.h`) and verified byte-for-byte against
/// `Ternary-Bonsai-2-27B-PQ2_0.gguf`.
///
/// SoA (output) layout is **identical to the TQ2 one** —
/// `[all d: N × 2 bytes FP16 LE][all qs: N × 32 bytes]` — so a `PQ2_0` weight
/// differs from a `TQ2_0_g128` weight only in its *decode table*
/// (`0b11 → +2.0` instead of `0b11 → 0.0`, see `decode_pq2` in
/// `kernel_sources/decode_ternary.rs`), not in the buffer the kernel reads.
///
/// Total size is unchanged: N × 34 bytes.
/// Returns `None` if the input is empty or its length is not a multiple of 34.
pub(super) fn reformat_pq2_aos_to_soa(aos_bytes: &[u8]) -> Option<Vec<u8>> {
    if aos_bytes.is_empty() || !aos_bytes.len().is_multiple_of(Q2_BLOCK_BYTES) {
        return None;
    }

    let n_blocks = aos_bytes.len() / Q2_BLOCK_BYTES;
    let mut soa = vec![0u8; n_blocks * Q2_BLOCK_BYTES];

    let (scales_section, data_section) = soa.split_at_mut(n_blocks * Q2_SCALE_BYTES);

    for i in 0..n_blocks {
        let block_start = i * Q2_BLOCK_BYTES;
        // PQ2 AoS: first 2 bytes are the scale, the remaining 32 are qs.
        scales_section[i * Q2_SCALE_BYTES..i * Q2_SCALE_BYTES + Q2_SCALE_BYTES]
            .copy_from_slice(&aos_bytes[block_start..block_start + Q2_SCALE_BYTES]);
        data_section[i * Q2_QS_BYTES..i * Q2_QS_BYTES + Q2_QS_BYTES].copy_from_slice(
            &aos_bytes[block_start + Q2_SCALE_BYTES..block_start + Q2_BLOCK_BYTES],
        );
    }

    Some(soa)
}

// ═══════════════════════════════════════════════════════════════════════════
// Ternary (TQ2_0_g128) reserved-code validation
// ═══════════════════════════════════════════════════════════════════════════

/// First occurrence of the reserved 2-bit code `0b11` inside a tensor that was
/// declared ternary (`TQ2_0_g128`).
///
/// Ternary data uses only `0b00 → −1`, `0b01 → 0`, `0b10 → +1`; every GPU and
/// CPU decoder in this workspace maps `0b11 → 0.0` (see `decode_tq2`). `PQ2_0`
/// instead defines `0b11 → +2`, so a `0b11` in a "ternary" tensor is the
/// signature of a `PQ2_0` tensor routed into the ternary path — the exact
/// silent corruption the mirror-image block layouts make possible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ReservedTernaryCode {
    /// Index of the 34-byte block containing the reserved code.
    pub block_index: usize,
    /// Index of the offending byte within the block's 32 `qs` bytes.
    pub byte_index: usize,
    /// LSB-first 2-bit lane (`0..=3`) inside that byte.
    pub lane: usize,
    /// Raw value of the offending byte (for diagnostics).
    pub byte_value: u8,
    /// Index of the offending weight within the whole tensor.
    pub weight_index: usize,
}

impl fmt::Display for ReservedTernaryCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "reserved 2-bit code 0b11 in a tensor declared ternary (TQ2_0_g128): \
             block {}, qs byte {} (0x{:02x}), lane {}, weight index {} — this is the \
             PQ2_0 (+2) encoding; upload PQ2_0 tensors via `upload_pq2_weight_soa`",
            self.block_index, self.byte_index, self.byte_value, self.lane, self.weight_index
        )
    }
}

/// Mask selecting the low bit of each LSB-first 2-bit lane in a `u64`.
const Q2_LANE_LOW_BITS: u64 = 0x5555_5555_5555_5555;

/// Reject any reserved `0b11` code in AoS `TQ2_0_g128` (qs-first) block bytes.
///
/// Scans only the 32 `qs` bytes of each 34-byte block — the `f16` scale bytes
/// are skipped, since an ordinary scale such as `1.0` (`0x3C00`) contains the
/// bit pattern `0b11` and would otherwise false-positive.
///
/// Runs at memory bandwidth: each block is four `u64` loads and the test
/// `x & (x >> 1) & 0x5555…` is exact — for every even bit position `p` (the low
/// bit of a lane) it keeps the bit iff lane bits `p` and `p+1` are both set, and
/// `p + 1 ≤ 7` always stays inside the same byte, so no lane straddles a byte.
///
/// Returns `Ok(())` when the input is clean (including for an empty input) or
/// when the length is not a multiple of 34 — a malformed length is the
/// reformatter's error to report, not this function's.
pub(super) fn validate_tq2_ternary_codes(aos_bytes: &[u8]) -> Result<(), ReservedTernaryCode> {
    for (block_index, block) in aos_bytes.chunks_exact(Q2_BLOCK_BYTES).enumerate() {
        for (word_index, chunk) in block[..Q2_QS_BYTES].chunks_exact(8).enumerate() {
            let mut word_bytes = [0u8; 8];
            word_bytes.copy_from_slice(chunk);
            let word = u64::from_le_bytes(word_bytes);
            let reserved = word & (word >> 1) & Q2_LANE_LOW_BITS;
            if reserved == 0 {
                continue;
            }
            let bit = reserved.trailing_zeros() as usize;
            let byte_index = word_index * 8 + bit / 8;
            let lane = (bit % 8) / 2;
            return Err(ReservedTernaryCode {
                block_index,
                byte_index,
                lane,
                byte_value: chunk[bit / 8],
                weight_index: block_index * Q2_BLOCK_WEIGHTS + byte_index * 4 + lane,
            });
        }
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Build `n_blocks` synthetic 34-byte blocks.
    ///
    /// `scale_of(i)` supplies the raw `f16` bits of block `i`'s scale and
    /// `qs_of(i, j)` the value of its `j`-th `qs` byte. `d_first` selects the
    /// `PQ2_0` (`d` at `0..2`) or `TQ2_0_g128` (`d` at `32..34`) order.
    fn build_blocks(
        n_blocks: usize,
        d_first: bool,
        scale_of: impl Fn(usize) -> u16,
        qs_of: impl Fn(usize, usize) -> u8,
    ) -> Vec<u8> {
        let mut out = vec![0u8; n_blocks * Q2_BLOCK_BYTES];
        for i in 0..n_blocks {
            let base = i * Q2_BLOCK_BYTES;
            let scale = scale_of(i).to_le_bytes();
            let (scale_at, qs_at) = if d_first {
                (base, base + Q2_SCALE_BYTES)
            } else {
                (base + Q2_QS_BYTES, base)
            };
            out[scale_at..scale_at + Q2_SCALE_BYTES].copy_from_slice(&scale);
            for j in 0..Q2_QS_BYTES {
                out[qs_at + j] = qs_of(i, j);
            }
        }
        out
    }

    /// Reconstruct the AoS bytes from a SoA buffer (inverse of the reformatters).
    fn soa_to_aos(soa: &[u8], d_first: bool) -> Vec<u8> {
        let n_blocks = soa.len() / Q2_BLOCK_BYTES;
        let (scales, data) = soa.split_at(n_blocks * Q2_SCALE_BYTES);
        let mut aos = vec![0u8; soa.len()];
        for i in 0..n_blocks {
            let base = i * Q2_BLOCK_BYTES;
            let (scale_at, qs_at) = if d_first {
                (base, base + Q2_SCALE_BYTES)
            } else {
                (base + Q2_QS_BYTES, base)
            };
            aos[scale_at..scale_at + Q2_SCALE_BYTES]
                .copy_from_slice(&scales[i * Q2_SCALE_BYTES..(i + 1) * Q2_SCALE_BYTES]);
            aos[qs_at..qs_at + Q2_QS_BYTES]
                .copy_from_slice(&data[i * Q2_QS_BYTES..(i + 1) * Q2_QS_BYTES]);
        }
        aos
    }

    #[test]
    fn pq2_reformat_round_trips_hand_built_blocks_bit_exactly() {
        // Scales taken from the real `Ternary-Bonsai-2-27B-PQ2_0.gguf`
        // (`output.weight`, first blocks — verified 2026-09-22 by reading the
        // raw bytes at the tensor's data offset, 11120992, in the actual
        // model file): 0.016724, 0.016434, 0.017609. (REQUIRED #4,
        // waves 1+1.5 gatekeeper review: the constants below previously read
        // [0x2247, 0x2235, 0x2283], which decode to 0.01226/0.01212/0.01272
        // — not the values this comment claims and not what the real file
        // contains at that offset; 0x2448/0x2435/0x2482 are the bytes
        // actually there.)
        let scales: [u16; 3] = [0x2448, 0x2435, 0x2482];
        let aos = build_blocks(
            3,
            true,
            |i| scales[i],
            |i, j| {
                (i as u8)
                    .wrapping_mul(37)
                    .wrapping_add(j as u8)
                    .wrapping_mul(3)
                    & 0xAA
            },
        );
        let soa = reformat_pq2_aos_to_soa(&aos).expect("pq2 reformat must accept 3 × 34 bytes");
        assert_eq!(soa.len(), aos.len(), "SoA must be the same size as AoS");

        // Scales section: the *first* two bytes of every block, in block order.
        for (i, expected) in scales.iter().enumerate() {
            let got = u16::from_le_bytes([soa[i * 2], soa[i * 2 + 1]]);
            assert_eq!(
                got, *expected,
                "block {i} scale must come from AoS bytes 0..2"
            );
        }
        // Data section: bytes 2..34 of every block, in block order.
        let data = &soa[3 * Q2_SCALE_BYTES..];
        for i in 0..3 {
            let src = &aos[i * Q2_BLOCK_BYTES + Q2_SCALE_BYTES..(i + 1) * Q2_BLOCK_BYTES];
            assert_eq!(&data[i * Q2_QS_BYTES..(i + 1) * Q2_QS_BYTES], src);
        }
        // Bit-exact round trip.
        assert_eq!(soa_to_aos(&soa, true), aos);
    }

    #[test]
    fn pq2_and_tq2_reformatters_are_mirror_images() {
        // One block whose scale (d-first) is 1.0 and whose qs bytes are 1..=32.
        let aos = build_blocks(1, true, |_| 0x3C00, |_, j| (j + 1) as u8);
        let pq2 = reformat_pq2_aos_to_soa(&aos).expect("pq2 reformat");
        let tq2 = reformat_tq2_aos_to_soa(&aos).expect("tq2 reformat accepts the same length");

        // PQ2 picks up the real scale…
        assert_eq!(u16::from_le_bytes([pq2[0], pq2[1]]), 0x3C00);
        // …while TQ2 reads the *last* two bytes (qs[30], qs[31] = 31, 32) as the
        // scale and the leading `d` as quant data — the silent MET-11 corruption.
        assert_eq!(
            u16::from_le_bytes([tq2[0], tq2[1]]),
            u16::from_le_bytes([31, 32])
        );
        assert_ne!(pq2, tq2, "the two layouts must not produce the same buffer");
    }

    #[test]
    fn tq2_reformat_still_round_trips_qs_first_blocks() {
        let aos = build_blocks(2, false, |i| 0x3C00 + i as u16, |i, j| (i * 32 + j) as u8);
        let soa = reformat_tq2_aos_to_soa(&aos).expect("tq2 reformat");
        assert_eq!(soa_to_aos(&soa, false), aos);
    }

    #[test]
    fn pq2_reformat_rejects_misaligned_and_empty_input() {
        assert!(reformat_pq2_aos_to_soa(&[]).is_none());
        assert!(reformat_pq2_aos_to_soa(&[0u8; Q2_BLOCK_BYTES - 1]).is_none());
        assert!(reformat_pq2_aos_to_soa(&[0u8; Q2_BLOCK_BYTES + 1]).is_none());
        assert!(reformat_pq2_aos_to_soa(&[0u8; Q2_BLOCK_BYTES * 2]).is_some());
    }

    #[test]
    fn validate_accepts_pure_ternary_codes() {
        // 0b10_01_00_10 and friends: no lane is 0b11.
        let aos = build_blocks(4, false, |_| 0x3C00, |_, j| [0x00, 0x24, 0x92, 0x48][j % 4]);
        assert_eq!(validate_tq2_ternary_codes(&aos), Ok(()));
        assert_eq!(validate_tq2_ternary_codes(&[]), Ok(()));
    }

    #[test]
    fn validate_ignores_scale_bytes_that_contain_the_reserved_pattern() {
        // f16 1.0 = 0x3C00 → byte 0x3C = 0b0011_1100 has two 0b11 lanes; the
        // scale must never be scanned or every real weight would be rejected.
        let aos = build_blocks(3, false, |_| 0x3C00, |_, _| 0x00);
        assert_eq!(validate_tq2_ternary_codes(&aos), Ok(()));
    }

    #[test]
    fn validate_reports_the_first_reserved_code_with_exact_coordinates() {
        // 0xE4 = 0b11_10_01_00 → lanes 0,1,2 = 00,01,10 and lane 3 = 0b11.
        let mut aos = build_blocks(2, false, |_| 0x3C00, |_, _| 0x00);
        aos[Q2_BLOCK_BYTES + 5] = 0xE4;
        let err = validate_tq2_ternary_codes(&aos).expect_err("0b11 must be rejected");
        assert_eq!(err.block_index, 1);
        assert_eq!(err.byte_index, 5);
        assert_eq!(err.lane, 3);
        assert_eq!(err.byte_value, 0xE4);
        assert_eq!(err.weight_index, Q2_BLOCK_WEIGHTS + 5 * 4 + 3);
        assert!(err.to_string().contains("upload_pq2_weight_soa"));
    }

    #[test]
    fn validate_finds_a_reserved_code_in_every_lane_and_byte_position() {
        for byte_index in 0..Q2_QS_BYTES {
            for lane in 0..4 {
                let mut aos = build_blocks(1, false, |_| 0x0000, |_, _| 0x00);
                aos[byte_index] = 0b11 << (lane * 2);
                let err = validate_tq2_ternary_codes(&aos)
                    .expect_err("a single 0b11 lane must be detected");
                assert_eq!(
                    (err.block_index, err.byte_index, err.lane),
                    (0, byte_index, lane)
                );
            }
        }
    }

    #[test]
    fn validate_skips_a_trailing_partial_block() {
        // A malformed length is the reformatter's error to report; the validator
        // must not index out of bounds on the ragged tail.
        let mut aos = build_blocks(1, false, |_| 0x0000, |_, _| 0x00);
        aos.extend_from_slice(&[0xFF; Q2_BLOCK_BYTES - 1]);
        assert_eq!(validate_tq2_ternary_codes(&aos), Ok(()));
        assert!(reformat_tq2_aos_to_soa(&aos).is_none());
    }
}
