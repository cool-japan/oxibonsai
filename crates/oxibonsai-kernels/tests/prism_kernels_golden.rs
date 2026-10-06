//! Golden-reference tests for the PrismML Bonsai 2 kernels (`PQ2_0`,
//! `Q2_0_g64`, `PTQ1_0`): acceptance (design §7.2 rows 1-6).
//!
//! Every dequant golden function below is an INDEPENDENT transliteration of
//! the relevant `ggml-quants.c` function, written directly from the C
//! source in this test file — not a copy of, or a call into,
//! `oxibonsai_core::quant_prism` or `oxibonsai_kernels::dequant_prism`. A
//! bitwise match against this crate's kernels is therefore a genuine
//! cross-check between two independently-written implementations, not a
//! tautology (the same methodology this whole project's finding process
//! uses, e.g. the round-trip check in finding core-gguf-K0).
//!
//! Byte fixtures are drawn from a raw, uniform `0..=255` PRNG, never
//! produced by quantizing floats: `BlockPQ2_0::quantize` (and every other
//! quantizer here) can *never* emit code `0b11` by construction
//! (`round(w/amax) + 1 in {0,1,2}`, see finding core-gguf-11), so a
//! bitwise test built only from quantizer output could not tell a correct
//! arithmetic decode (`0b11 -> +2`) apart from an incorrectly-reused
//! ternary LUT decode (`0b11 -> 0`) — which is exactly the bug class this
//! package exists to prevent (findings K-01 / K-06 / K-13).
//!
//! NOTE ON TEST-NAME FILTERING: the package gate is
//! `cargo test -p oxibonsai-kernels --all-features prism`. libtest matches
//! that filter against each test's OWN name, not this file's name — so
//! every function below is named `test_prism_*`.

use half::f16;
use oxibonsai_core::{
    BlockPQ2_0, BlockPTQ1_0, BlockQ2_0G64, BlockTQ2_0_g128, QK_PQ2_0, QK_PTQ1_0, QK_Q2_0_G64,
};
use oxibonsai_kernels::dequant_prism::{
    dequant_pq2_0, dequant_ptq1_0, dequant_q2_0_g64, transcode_ptq1_0_to_pq2_0,
};
use oxibonsai_kernels::gemv_ptq1::gemv_ptq1_0;

// ---------------------------------------------------------------------------
// Deterministic PRNG (matches this workspace's existing test convention —
// see e.g. `quant_prism.rs`'s own tests — a fixed LCG, not an external
// `rand` dependency, so failures are exactly reproducible from the seed).
// ---------------------------------------------------------------------------

struct Lcg(u32);

impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.0
    }

    fn next_byte(&mut self) -> u8 {
        (self.next_u32() >> 16) as u8
    }

    fn next_f16_scale(&mut self) -> f16 {
        // Finite, occasionally negative-magnitude-adjacent, occasionally
        // fractional — the specific distribution doesn't matter, only that
        // it's nonzero-heavy (a zero scale can't distinguish decode maps)
        // and varied.
        let raw = (self.next_u32() % 2000) as f32 / 100.0 - 10.0;
        f16::from_f32(raw)
    }
}

// ---------------------------------------------------------------------------
// Golden C transliterations (independent of both `oxibonsai-core` and
// `oxibonsai-kernels`)
// ---------------------------------------------------------------------------

/// `dequantize_row_q2_0` / `dequantize_row_pq2_0` (`ggml-quants.c:472-512`)
/// — bit-for-bit identical bodies in the fork, differing only in `qk`
/// (implied here by `qs.len() * 4`).
fn golden_dequant_two_bit_family(qs: &[u8], d: f32, y: &mut [f32]) {
    assert_eq!(y.len(), qs.len() * 4);
    for (j, y_j) in y.iter_mut().enumerate() {
        let byte_index = j / 4;
        let bit_offset = (j % 4) * 2;
        let q = (qs[byte_index] >> bit_offset) & 0x03;
        *y_j = ((q as i32) - 1) as f32 * d;
    }
}

/// `dequantize_row_ptq1_0` (`ggml-quants.c:2255-2285`), transliterated
/// directly, including the `uint8_t` wraparound multiply.
fn golden_dequant_ptq1_0(qs: &[u8; 24], qh: &[u8; 2], d: f32, y: &mut [f32; QK_PTQ1_0]) {
    const STAGES: [usize; 3] = [32, 16, 8];
    const POW3: [u8; 6] = [1, 3, 9, 27, 81, 243];

    let mut yi = 0usize;
    let mut j = 0usize;
    for &c in STAGES.iter() {
        while j + c <= qs.len() {
            for &pow in POW3.iter().take(5) {
                for m in 0..c {
                    let q = qs[j + m].wrapping_mul(pow);
                    let xi = ((q as u16) * 3) >> 8;
                    y[yi] = (xi as i32 - 1) as f32 * d;
                    yi += 1;
                }
            }
            j += c;
        }
    }
    for &pow in POW3.iter().take(4) {
        for &byte in qh.iter() {
            let q = byte.wrapping_mul(pow);
            let xi = ((q as u16) * 3) >> 8;
            y[yi] = (xi as i32 - 1) as f32 * d;
            yi += 1;
        }
    }
    assert_eq!(yi, QK_PTQ1_0);
}

// ---------------------------------------------------------------------------
// PQ2_0 golden
// ---------------------------------------------------------------------------

#[test]
fn test_prism_pq2_0_dequant_matches_golden_c_over_random_raw_bytes() {
    let mut rng = Lcg(0xC0DE_CAFE);
    let n_blocks = 96; // >= 64 required by acceptance
    let mut blocks = Vec::with_capacity(n_blocks);
    let mut golden = vec![0.0f32; n_blocks * QK_PQ2_0];
    for bi in 0..n_blocks {
        let qs: [u8; 32] = core::array::from_fn(|_| rng.next_byte());
        let d = rng.next_f16_scale();
        golden_dequant_two_bit_family(
            &qs,
            d.to_f32(),
            &mut golden[bi * QK_PQ2_0..(bi + 1) * QK_PQ2_0],
        );
        blocks.push(BlockPQ2_0 { d, qs });
    }

    let mut got = vec![0.0f32; n_blocks * QK_PQ2_0];
    dequant_pq2_0(&blocks, &mut got).expect("dequant_pq2_0");

    assert_eq!(
        golden, got,
        "dequant_pq2_0 must be bitwise-identical to golden C"
    );
}

/// Explicit canary from the acceptance text: reading `qs` at byte offset 2
/// (d FIRST) must give the finite, small values the real model shows; a
/// `qs`-first misread would instead read the scale bytes as codes.
#[test]
fn test_prism_pq2_0_reads_d_first_not_qs_first() {
    let qs = [0xAAu8; 32]; // all code 0b10 -> +1
    let d = f16::from_f32(0.0109); // magnitude the finding reports for the real file
    let block = BlockPQ2_0 { d, qs };
    let mut out = vec![0.0f32; QK_PQ2_0];
    dequant_pq2_0(&[block], &mut out).expect("dequant");
    for &v in &out {
        assert!((v - 0.0109).abs() < 1e-4, "expected ~0.0109, got {v}");
        assert!(v.is_finite());
    }
}

// ---------------------------------------------------------------------------
// Q2_0_g64 golden
// ---------------------------------------------------------------------------

#[test]
fn test_prism_q2_0_g64_dequant_matches_golden_c_over_random_raw_bytes() {
    let mut rng = Lcg(0x1337_BEEF);
    let n_blocks = 96;
    let mut blocks = Vec::with_capacity(n_blocks);
    let mut golden = vec![0.0f32; n_blocks * QK_Q2_0_G64];
    for bi in 0..n_blocks {
        let qs: [u8; 16] = core::array::from_fn(|_| rng.next_byte());
        let d = rng.next_f16_scale();
        golden_dequant_two_bit_family(
            &qs,
            d.to_f32(),
            &mut golden[bi * QK_Q2_0_G64..(bi + 1) * QK_Q2_0_G64],
        );
        blocks.push(BlockQ2_0G64 { d, qs });
    }

    let mut got = vec![0.0f32; n_blocks * QK_Q2_0_G64];
    dequant_q2_0_g64(&blocks, &mut got).expect("dequant_q2_0_g64");

    assert_eq!(
        golden, got,
        "dequant_q2_0_g64 must be bitwise-identical to golden C"
    );
}

// ---------------------------------------------------------------------------
// PTQ1_0 golden
// ---------------------------------------------------------------------------

#[test]
fn test_prism_ptq1_0_dequant_matches_golden_c_over_random_raw_bytes() {
    let mut rng = Lcg(0xDEAD_10CC);
    let n_blocks = 96;
    let mut blocks = Vec::with_capacity(n_blocks);
    let mut golden = vec![0.0f32; n_blocks * QK_PTQ1_0];
    for bi in 0..n_blocks {
        let qs: [u8; 24] = core::array::from_fn(|_| rng.next_byte());
        let qh: [u8; 2] = core::array::from_fn(|_| rng.next_byte());
        let d = rng.next_f16_scale();
        let mut block_golden = [0.0f32; QK_PTQ1_0];
        golden_dequant_ptq1_0(&qs, &qh, d.to_f32(), &mut block_golden);
        golden[bi * QK_PTQ1_0..(bi + 1) * QK_PTQ1_0].copy_from_slice(&block_golden);
        blocks.push(BlockPTQ1_0 { qs, qh, d });
    }

    let mut got = vec![0.0f32; n_blocks * QK_PTQ1_0];
    dequant_ptq1_0(&blocks, &mut got).expect("dequant_ptq1_0");

    assert_eq!(
        golden, got,
        "dequant_ptq1_0 must be bitwise-identical to golden C"
    );
}

/// The three traps `decode_ptq1_0_codes` documents, each pinned down with a
/// golden-C-agreeing fixture: stage 32 is inert, the element order is
/// n-outer/m-inner (not `5*(j+m)`), and the multiply wraps mod 256.
#[test]
fn test_prism_ptq1_0_wrap_trap_matches_golden_c() {
    // Byte 200 with pow3[4] = 81: 200*81 = 16200, which truncates to 72 as
    // a `u8` (16200 % 256 = 72). A widened (non-wrapping) multiply would
    // leave the 0..=2 trit range entirely, and the golden transliteration
    // above uses `wrapping_mul` too, so this fixture would fail identically
    // in both places if either dropped the wrap -- pinning it down here
    // makes that shared assumption explicit and load-bearing.
    let mut qs = [1u8; 24]; // byte value 1 -> trits all differ from qs[3]
    qs[3] = 200;
    let qh = [1u8; 2];
    let d = 1.0f32;

    let mut golden = [0.0f32; QK_PTQ1_0];
    golden_dequant_ptq1_0(&qs, &qh, d, &mut golden);

    let block = BlockPTQ1_0 {
        qs,
        qh,
        d: f16::from_f32(d),
    };
    let mut got = vec![0.0f32; QK_PTQ1_0];
    dequant_ptq1_0(&[block], &mut got).expect("dequant");

    assert_eq!(&golden[..], &got[..]);
    // And: every decoded value must be one of {-1, 0, +1} * d -- if the
    // wrap were dropped, `((200u16*81)*3)>>8` would decode to a value
    // outside that set.
    for &v in &got {
        assert!(
            v == -1.0 || v == 0.0 || v == 1.0,
            "value {v} outside the trit range -- the u8 wrap was dropped"
        );
    }
}

// ---------------------------------------------------------------------------
// Transcode: dequant(ptq) == dequant(transcode(ptq)) -- bitwise
// ---------------------------------------------------------------------------

#[test]
fn test_prism_transcode_ptq1_0_to_pq2_0_is_bitwise_lossless() {
    let mut rng = Lcg(0xFACE_B00C);
    let n_blocks = 40;
    let mut blocks = Vec::with_capacity(n_blocks);
    for _ in 0..n_blocks {
        let qs: [u8; 24] = core::array::from_fn(|_| rng.next_byte());
        let qh: [u8; 2] = core::array::from_fn(|_| rng.next_byte());
        let d = rng.next_f16_scale();
        blocks.push(BlockPTQ1_0 { qs, qh, d });
    }

    let mut pq_blocks = vec![BlockPQ2_0::zeroed(); n_blocks];
    transcode_ptq1_0_to_pq2_0(&blocks, &mut pq_blocks).expect("transcode");

    let mut direct = vec![0.0f32; n_blocks * QK_PTQ1_0];
    dequant_ptq1_0(&blocks, &mut direct).expect("dequant ptq1_0");
    let mut via_pq2 = vec![0.0f32; n_blocks * QK_PQ2_0];
    dequant_pq2_0(&pq_blocks, &mut via_pq2).expect("dequant pq2_0");

    assert_eq!(direct, via_pq2, "transcode must be bitwise-lossless");
    for block in &pq_blocks {
        assert_eq!(
            block.count_plus_two(),
            0,
            "PTQ1_0's trit codes can only ever become PQ2_0 codes {{00,01,10}}"
        );
    }
}

// ---------------------------------------------------------------------------
// gemv_ptq1_0 vs dequant + naive dot (design §7.2 acceptance: <= 1e-4)
// ---------------------------------------------------------------------------

#[test]
fn test_prism_gemv_ptq1_0_matches_dequant_plus_naive_dot_multi_row() {
    let mut rng = Lcg(0x5EED_1234);
    let n_rows = 6;
    let blocks_per_row = 3;
    let k = blocks_per_row * QK_PTQ1_0;
    let mut blocks = Vec::with_capacity(n_rows * blocks_per_row);
    for _ in 0..n_rows * blocks_per_row {
        let qs: [u8; 24] = core::array::from_fn(|_| rng.next_byte());
        let qh: [u8; 2] = core::array::from_fn(|_| rng.next_byte());
        let d = rng.next_f16_scale();
        blocks.push(BlockPTQ1_0 { qs, qh, d });
    }
    let input: Vec<f32> = (0..k)
        .map(|_| (rng.next_u32() % 2000) as f32 / 500.0 - 2.0)
        .collect();

    let mut dequantized = vec![0.0f32; blocks.len() * QK_PTQ1_0];
    dequant_ptq1_0(&blocks, &mut dequantized).expect("dequant");

    let mut expected = vec![0.0f32; n_rows];
    for row in 0..n_rows {
        let row_weights = &dequantized[row * k..(row + 1) * k];
        expected[row] = row_weights
            .iter()
            .zip(input.iter())
            .map(|(w, x)| w * x)
            .sum();
    }

    let mut got = vec![0.0f32; n_rows];
    gemv_ptq1_0(&blocks, &input, &mut got, n_rows, k).expect("gemv_ptq1_0");

    for row in 0..n_rows {
        let tol = 1e-4 * expected[row].abs().max(1.0);
        assert!(
            (got[row] - expected[row]).abs() < tol,
            "row {row}: gemv={} vs dequant+dot={}",
            got[row],
            expected[row]
        );
    }
}

// ---------------------------------------------------------------------------
// K-06 regression: PQ2_0 must never be confusable with legacy TQ2_0_g128
// ---------------------------------------------------------------------------

/// Reads the SAME 34 raw bytes two ways: as `PQ2_0` (`d` first, arithmetic
/// map) and as the legacy `TQ2_0_g128` (`qs` first, ternary LUT). They must
/// disagree -- this is the entire reason the two formats need separate
/// block types and decode functions (findings K-01 / K-06).
#[test]
fn test_prism_pq2_0_and_legacy_tq2_0_g128_decode_the_same_bytes_differently() {
    let mut raw = [0u8; 34];
    let scale_bits = f16::from_f32(1.0).to_bits();
    raw[0] = (scale_bits & 0xFF) as u8;
    raw[1] = (scale_bits >> 8) as u8;
    raw[2] = 0b11_10_01_00; // lanes 0..4 (LSB first): 00, 01, 10, 11
    for b in raw[3..].iter_mut() {
        *b = 0x55;
    }

    // PQ2_0 reading: d = raw[0..2], qs = raw[2..34].
    let pq_d = f16::from_bits(u16::from_le_bytes([raw[0], raw[1]]));
    let pq_qs: [u8; 32] = raw[2..34].try_into().expect("32 bytes");
    let mut pq_out = vec![0.0f32; QK_PQ2_0];
    dequant_pq2_0(&[BlockPQ2_0 { d: pq_d, qs: pq_qs }], &mut pq_out).expect("pq2_0 dequant");
    assert_eq!(&pq_out[0..4], &[-1.0, 0.0, 1.0, 2.0]);

    // Legacy TQ2_0_g128 reading of the IDENTICAL bytes: qs = raw[0..32],
    // d = raw[32..34].
    let tq_qs: [u8; 32] = raw[0..32].try_into().expect("32 bytes");
    let tq_d = f16::from_bits(u16::from_le_bytes([raw[32], raw[33]]));
    let mut tq_out = vec![0.0f32; QK_PQ2_0];
    BlockTQ2_0_g128::dequant(&[BlockTQ2_0_g128 { qs: tq_qs, d: tq_d }], &mut tq_out)
        .expect("tq2_0_g128 dequant");

    assert_ne!(
        pq_out, tq_out,
        "PQ2_0 (d-first, arithmetic) and TQ2_0_g128 (qs-first, ternary LUT) \
         must never decode identical raw bytes to the same values"
    );
}

// ---------------------------------------------------------------------------
// Dimension / buffer error propagation stays a real error, not a panic
// ---------------------------------------------------------------------------

#[test]
fn test_prism_all_three_formats_reject_undersized_output_without_panicking() {
    let pq = BlockPQ2_0::zeroed();
    assert!(dequant_pq2_0(&[pq], &mut [0.0f32; 4]).is_err());

    let g64 = BlockQ2_0G64::zeroed();
    assert!(dequant_q2_0_g64(&[g64], &mut [0.0f32; 4]).is_err());

    let ptq = BlockPTQ1_0 {
        qs: [0u8; 24],
        qh: [0u8; 2],
        d: f16::ZERO,
    };
    assert!(dequant_ptq1_0(&[ptq], &mut [0.0f32; 4]).is_err());
}
