//! Q1 / TQ2 weight block reformatters (AoS → SoA).
//!
//! The Metal MSL kernels expect a Structure-of-Arrays layout where all scales
//! come first and all packed quant data follows, in order to maximize
//! coalesced reads.  These helpers translate the on-disk Array-of-Structures
//! layout into the SoA layout consumed by the GPU.

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

    if aos_bytes.is_empty() || aos_bytes.len() % BLOCK_SIZE != 0 {
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

/// Reformat TQ2_0_g128 AoS → SoA for the Metal ternary GEMV kernel.
///
/// AoS (input) block layout: `{ qs: [u8; 32], d: f16 }` = 34 bytes
/// SoA (output) layout: `[all d: N × 2 bytes FP16 LE][all qs: N × 32 bytes]`
/// Note: the ternary MSL kernel consumes this layout directly — scales first,
/// then qs data, matching the convention in `scirs2_backend::upload_weights_ternary`.
///
/// Total size is unchanged: N × 34 bytes.
/// Returns `None` if the input length is not a multiple of 34.
pub(super) fn reformat_tq2_aos_to_soa(aos_bytes: &[u8]) -> Option<Vec<u8>> {
    const BLOCK_SIZE: usize = 34;
    const SCALE_SIZE: usize = 2;
    const DATA_SIZE: usize = 32;

    if aos_bytes.is_empty() || aos_bytes.len() % BLOCK_SIZE != 0 {
        return None;
    }

    let n_blocks = aos_bytes.len() / BLOCK_SIZE;
    let mut soa = vec![0u8; n_blocks * BLOCK_SIZE];

    let (scales_section, data_section) = soa.split_at_mut(n_blocks * SCALE_SIZE);

    for i in 0..n_blocks {
        let block_start = i * BLOCK_SIZE;
        // TQ2 AoS: first 32 bytes are qs, last 2 are scale. SoA: scales first, then qs.
        data_section[i * DATA_SIZE..i * DATA_SIZE + DATA_SIZE]
            .copy_from_slice(&aos_bytes[block_start..block_start + DATA_SIZE]);
        scales_section[i * SCALE_SIZE..i * SCALE_SIZE + SCALE_SIZE]
            .copy_from_slice(&aos_bytes[block_start + DATA_SIZE..block_start + BLOCK_SIZE]);
    }

    Some(soa)
}
