//! Byte-exactness regression gate for the K-quant and Q4_0 codecs.
//!
//! Every reference decoder in [`ggml_ref`] is an **independent** transliteration
//! of `ggml/src/ggml-quants.c`, written against the C source and operating on
//! raw block *bytes*. Nothing in that module calls into `oxibonsai_core`, so a
//! transliteration mistake in `src/quant_k.rs` cannot hide behind a matching
//! mistake here.
//!
//! The suite has four layers:
//!
//! 1. **Hand-computed sparse goldens** — a block whose every output element is
//!    derived by hand in the comments, asserted exactly. These pin the
//!    sub-block/nibble *ordering*, which is what was wrong before
//!    (`core-gguf-K0`): Q2_K's `is++` / shift 0,2,4,6 stepping, Q4_K's
//!    "32 low nibbles then 32 high nibbles", Q6_K's four-lane interleave with
//!    `sc[is+0,+2,+4,+6]`, Q3_K's `-32` bias and inverted `hmask`, and Q5_K's
//!    `u1/u2 <<= 2` masks.
//! 2. **Dense goldens** — pseudo-random block bytes decoded by both the crate
//!    and the reference, compared bit-exactly (`to_bits()`).
//! 3. **Encoder -> ggml-decoder round trip** on a smooth 256-sample signal. The
//!    broken codecs scored 1.20-5.64 max error here; ggml-exact ones land in the
//!    0.02-0.21 self-consistent band.
//! 4. **Q4_0 / Q8_0 non-regression**, plus Q4_0 encoder bit-identity against a
//!    hand-computed ggml block whose extremum is *negative* (`core-gguf-Q40` —
//!    the case where the sign of `d = max / -8` matters).

use half::f16;

use oxibonsai_core::quant_k::{BlockQ2K, BlockQ3K, BlockQ4K, QK_K};
use oxibonsai_core::quant_k_ext::{BlockQ5K, BlockQ6K};
use oxibonsai_core::quant_std::{BlockQ4_0, BlockQ8_0, QK_Q4_0, QK_Q8_0};

// ===========================================================================
// Independent ggml transliteration (ggml/src/ggml-quants.c)
// ===========================================================================

mod ggml_ref {
    use half::f16;

    /// `GGML_FP16_TO_FP32` on a little-endian 2-byte field.
    fn fp16(bytes: &[u8]) -> f32 {
        f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32()
    }

    /// `ggml-quants.c:676` — round-half-to-even magic-number trick.
    pub fn nearest_int(fval: f32) -> i32 {
        let val = fval + 12_582_912.0_f32;
        let i = val.to_bits() as i32;
        (i & 0x007f_ffff) - 0x0040_0000
    }

    /// `ggml-quants.c:935`.
    fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
        if j < 4 {
            (q[j] & 63, q[j + 4] & 63)
        } else {
            (
                (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
                (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
            )
        }
    }

    /// `dequantize_row_q2_K` (`ggml-quants.c:1016`).
    ///
    /// Block bytes: `scales[16] | qs[64] | d:f16 | dmin:f16`.
    pub fn dequantize_row_q2_k(block: &[u8]) -> Vec<f32> {
        let scales = &block[0..16];
        let qs = &block[16..80];
        let d = fp16(&block[80..82]);
        let min = fp16(&block[82..84]);

        let mut y = Vec::with_capacity(256);
        let mut is = 0usize;
        let mut q = 0usize;
        let mut n = 0usize;
        while n < 256 {
            let mut shift = 0u32;
            for _j in 0..4 {
                let sc = scales[is];
                is += 1;
                let dl = d * ((sc & 0xF) as f32);
                let ml = min * ((sc >> 4) as f32);
                for l in 0..16usize {
                    y.push(dl * (((qs[q + l] >> shift) & 3) as f32) - ml);
                }
                let sc = scales[is];
                is += 1;
                let dl = d * ((sc & 0xF) as f32);
                let ml = min * ((sc >> 4) as f32);
                for l in 0..16usize {
                    y.push(dl * (((qs[q + l + 16] >> shift) & 3) as f32) - ml);
                }
                shift += 2;
            }
            q += 32;
            n += 128;
        }
        y
    }

    /// `dequantize_row_q3_K` (`ggml-quants.c:1360`).
    ///
    /// Block bytes: `hmask[32] | qs[64] | scales[12] | d:f16`.
    pub fn dequantize_row_q3_k(block: &[u8]) -> Vec<f32> {
        const KMASK1: u32 = 0x0303_0303;
        const KMASK2: u32 = 0x0f0f_0f0f;

        let hm = &block[0..32];
        let qs = &block[32..96];
        let raw = &block[96..108];
        let d_all = fp16(&block[108..110]);

        let mut aux = [0u32; 4];
        aux[0] = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
        aux[1] = u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]);
        aux[2] = u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]);
        let tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
        aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
        aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
        aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);

        let mut scales = [0i8; 16];
        for (k, word) in aux.iter().enumerate() {
            for (t, b) in word.to_le_bytes().iter().enumerate() {
                scales[4 * k + t] = *b as i8;
            }
        }

        let mut y = Vec::with_capacity(256);
        let mut is = 0usize;
        let mut q = 0usize;
        let mut m: u8 = 1;
        let mut n = 0usize;
        while n < 256 {
            let mut shift = 0u32;
            for _j in 0..4 {
                let dl = d_all * ((scales[is] as i32 - 32) as f32);
                is += 1;
                for l in 0..16usize {
                    let hb = if (hm[l] & m) != 0 { 0i32 } else { 4i32 };
                    y.push(dl * ((((qs[q + l] >> shift) & 3) as i32 - hb) as f32));
                }
                let dl = d_all * ((scales[is] as i32 - 32) as f32);
                is += 1;
                for l in 0..16usize {
                    let hb = if (hm[l + 16] & m) != 0 { 0i32 } else { 4i32 };
                    y.push(dl * ((((qs[q + l + 16] >> shift) & 3) as i32 - hb) as f32));
                }
                shift += 2;
                m <<= 1;
            }
            q += 32;
            n += 128;
        }
        y
    }

    /// `dequantize_row_q4_K` (`ggml-quants.c:1584`).
    ///
    /// Block bytes: `d:f16 | dmin:f16 | scales[12] | qs[128]`.
    pub fn dequantize_row_q4_k(block: &[u8]) -> Vec<f32> {
        let d = fp16(&block[0..2]);
        let min = fp16(&block[2..4]);
        let scales = &block[4..16];
        let qs = &block[16..144];

        let mut y = Vec::with_capacity(256);
        let mut q = 0usize;
        let mut is = 0usize;
        let mut j = 0usize;
        while j < 256 {
            let (sc, m) = get_scale_min_k4(is, scales);
            let d1 = d * (sc as f32);
            let m1 = min * (m as f32);
            let (sc, m) = get_scale_min_k4(is + 1, scales);
            let d2 = d * (sc as f32);
            let m2 = min * (m as f32);
            for l in 0..32usize {
                y.push(d1 * ((qs[q + l] & 0xF) as f32) - m1);
            }
            for l in 0..32usize {
                y.push(d2 * ((qs[q + l] >> 4) as f32) - m2);
            }
            q += 32;
            is += 2;
            j += 64;
        }
        y
    }

    /// `dequantize_row_q5_K` (`ggml-quants.c:1786`).
    ///
    /// Block bytes: `d:f16 | dmin:f16 | scales[12] | qh[32] | qs[128]`.
    pub fn dequantize_row_q5_k(block: &[u8]) -> Vec<f32> {
        let d = fp16(&block[0..2]);
        let min = fp16(&block[2..4]);
        let scales = &block[4..16];
        let qh = &block[16..48];
        let ql = &block[48..176];

        let mut y = Vec::with_capacity(256);
        let mut q = 0usize;
        let mut is = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        let mut j = 0usize;
        while j < 256 {
            let (sc, m) = get_scale_min_k4(is, scales);
            let d1 = d * (sc as f32);
            let m1 = min * (m as f32);
            let (sc, m) = get_scale_min_k4(is + 1, scales);
            let d2 = d * (sc as f32);
            let m2 = min * (m as f32);
            for l in 0..32usize {
                let hi = if (qh[l] & u1) != 0 { 16u32 } else { 0 };
                y.push(d1 * (((ql[q + l] & 0xF) as u32 + hi) as f32) - m1);
            }
            for l in 0..32usize {
                let hi = if (qh[l] & u2) != 0 { 16u32 } else { 0 };
                y.push(d2 * (((ql[q + l] >> 4) as u32 + hi) as f32) - m2);
            }
            q += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
            j += 64;
        }
        y
    }

    /// `dequantize_row_q6_K` (`ggml-quants.c:1994`).
    ///
    /// Block bytes: `ql[128] | qh[64] | scales[16]:i8 | d:f16`.
    pub fn dequantize_row_q6_k(block: &[u8]) -> Vec<f32> {
        let ql_all = &block[0..128];
        let qh_all = &block[128..192];
        let sc_all = &block[192..208];
        let d = fp16(&block[208..210]);

        let mut y = vec![0.0f32; 256];
        let mut out = 0usize;
        let mut ql = 0usize;
        let mut qh = 0usize;
        let mut sc = 0usize;
        let mut n = 0usize;
        while n < 256 {
            for l in 0..32usize {
                let is = l / 16;
                let h = qh_all[qh + l];
                let q1 = (((ql_all[ql + l] & 0xF) | ((h & 3) << 4)) as i32) - 32;
                let q2 = (((ql_all[ql + l + 32] & 0xF) | (((h >> 2) & 3) << 4)) as i32) - 32;
                let q3 = (((ql_all[ql + l] >> 4) | (((h >> 4) & 3) << 4)) as i32) - 32;
                let q4 = (((ql_all[ql + l + 32] >> 4) | (((h >> 6) & 3) << 4)) as i32) - 32;
                y[out + l] = d * (sc_all[sc + is] as i8 as f32) * (q1 as f32);
                y[out + l + 32] = d * (sc_all[sc + is + 2] as i8 as f32) * (q2 as f32);
                y[out + l + 64] = d * (sc_all[sc + is + 4] as i8 as f32) * (q3 as f32);
                y[out + l + 96] = d * (sc_all[sc + is + 6] as i8 as f32) * (q4 as f32);
            }
            out += 128;
            ql += 64;
            qh += 32;
            sc += 8;
            n += 128;
        }
        y
    }

    /// `dequantize_row_q4_0` (`ggml-quants.c`): `d:f16 | qs[16]`, lo/hi split.
    pub fn dequantize_row_q4_0(block: &[u8]) -> Vec<f32> {
        let d = fp16(&block[0..2]);
        let qs = &block[2..18];
        let mut y = vec![0.0f32; 32];
        for j in 0..16usize {
            let x0 = (qs[j] & 0x0F) as i32 - 8;
            let x1 = (qs[j] >> 4) as i32 - 8;
            y[j] = (x0 as f32) * d;
            y[j + 16] = (x1 as f32) * d;
        }
        y
    }

    /// `dequantize_row_q8_0` (`ggml-quants.c`): `d:f16 | qs[32]:i8`.
    pub fn dequantize_row_q8_0(block: &[u8]) -> Vec<f32> {
        let d = fp16(&block[0..2]);
        let qs = &block[2..34];
        (0..32).map(|j| (qs[j] as i8 as f32) * d).collect()
    }

    /// `quantize_row_q4_0_ref` (`ggml-quants.c:148-178`), emitting the 18 block
    /// bytes in on-disk order.
    pub fn quantize_row_q4_0_ref(x: &[f32]) -> Vec<u8> {
        let qk = 32usize;
        let mut out = Vec::with_capacity(18 * (x.len() / qk));
        for i in 0..(x.len() / qk) {
            let mut amax = 0.0f32;
            let mut max = 0.0f32;
            for j in 0..qk {
                let v = x[i * qk + j];
                if amax < v.abs() {
                    amax = v.abs();
                    max = v;
                }
            }
            let d = max / -8.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            out.extend_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
            for j in 0..(qk / 2) {
                let x0 = x[i * qk + j] * id;
                let x1 = x[i * qk + qk / 2 + j] * id;
                let xi0 = ((x0 + 8.5) as i8).min(15) as u8;
                let xi1 = ((x1 + 8.5) as i8).min(15) as u8;
                out.push(xi0 | (xi1 << 4));
            }
        }
        out
    }
}

// ===========================================================================
// Helpers
// ===========================================================================

/// 8-byte-aligned byte buffer so `slice_from_bytes` can reinterpret it.
#[repr(C, align(8))]
struct Aligned<const N: usize>([u8; N]);

/// Deterministic byte stream (no `rand` dependency).
fn lcg_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as u8
        })
        .collect()
}

/// A smooth 256-sample signal in roughly `[-1, 1]` — the round-trip probe input.
///
/// One sine period per 256 samples plus a slow cosine, so consecutive samples
/// differ by ~0.025 and a 16-element Q2_K sub-block spans well under half the
/// full range. This is what "smooth" means for the self-consistency band: the
/// per-sub-block dynamic range, not the global one, sets the quantization step.
fn smooth_signal(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f32;
            0.85 * (t * std::f32::consts::TAU / 256.0).sin()
                + 0.15 * (t * std::f32::consts::TAU / 512.0).cos()
        })
        .collect()
}

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Assert two decodes agree on every bit (not merely "close").
fn assert_bit_exact(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    let mismatches: Vec<usize> = (0..got.len())
        .filter(|&i| got[i].to_bits() != want[i].to_bits())
        .collect();
    assert!(
        mismatches.is_empty(),
        "{label}: {}/{} elements differ from the ggml reference; first at {} \
         (got {:?}, want {:?})",
        mismatches.len(),
        got.len(),
        mismatches[0],
        got[mismatches[0]],
        want[mismatches[0]]
    );
}

/// Assert every element of `got` equals `want` exactly (hand-computed goldens).
fn assert_exact(label: &str, got: &[f32], want: &[f32]) {
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(g, w, "{label}: index {i}: got {g}, expected {w}");
    }
}

// ===========================================================================
// Q2_K
// ===========================================================================

/// Hand-computed Q2_K golden.
///
/// `d = 0.5`, `dmin = 0.25`, `scales[0] = 0x12` (sc = 2, mn = 1),
/// `scales[1] = 0x31` (sc = 1, mn = 3), every other scale byte zero;
/// `qs[0] = 0xE4` (2-bit lanes 0,1,2,3), `qs[16] = 0x1B` (lanes 3,2,1,0).
///
/// ggml walks `is = 0,1,2,…` while `shift` steps 0,2,4,6 per 128-element half,
/// so the **first** `j` iteration emits `y[0..16]` from `qs[0..16] >> 0` under
/// `scales[0]` and `y[16..32]` from `qs[16..32] >> 0` under `scales[1]`:
///
/// - `y[0]      = 0.5*2 * ((0xE4 >> 0) & 3) - 0.25*1 = 1.0*0 - 0.25 = -0.25`
/// - `y[1..16]  = 1.0*0 - 0.25                        = -0.25`
/// - `y[16]     = 0.5*1 * ((0x1B >> 0) & 3) - 0.25*3 = 0.5*3 - 0.75 =  0.75`
/// - `y[17..32] = 0.5*0 - 0.75                        = -0.75`
/// - `y[32..256]`: every remaining scale byte is 0, so `dl = ml = 0`.
///
/// The pre-fix element-sequential decoder produced `y[1] = +0.75` here (it read
/// lane 1 of `qs[0]` for element 1), so this golden is discriminating.
fn q2k_sparse_golden() -> Aligned<84> {
    let mut b = [0u8; 84];
    b[0] = 0x12;
    b[1] = 0x31;
    b[16] = 0xE4; // qs[0]
    b[32] = 0x1B; // qs[16]
    b[80..82].copy_from_slice(&f16::from_f32(0.5).to_bits().to_le_bytes());
    b[82..84].copy_from_slice(&f16::from_f32(0.25).to_bits().to_le_bytes());
    Aligned(b)
}

#[test]
fn q2k_hand_computed_golden_block() {
    let buf = q2k_sparse_golden();
    let blocks = BlockQ2K::slice_from_bytes(&buf.0).expect("aligned 84-byte block parses");
    assert_eq!(blocks.len(), 1);

    let mut got = vec![0.0f32; QK_K];
    BlockQ2K::dequant(blocks, &mut got).expect("dequant");

    let mut want = vec![0.0f32; QK_K];
    for w in want.iter_mut().take(16) {
        *w = -0.25;
    }
    want[16] = 0.75;
    for w in want.iter_mut().take(32).skip(17) {
        *w = -0.75;
    }
    assert_exact("Q2_K hand golden", &got, &want);

    // …and the independent reference decoder agrees on every bit.
    assert_bit_exact(
        "Q2_K hand golden vs ggml",
        &got,
        &ggml_ref::dequantize_row_q2_k(&buf.0),
    );
}

#[test]
fn q2k_dense_golden_matches_ggml_reference() {
    let rnd = lcg_bytes(80, 0x2ACE_0001);
    let mut b = [0u8; 84];
    b[0..80].copy_from_slice(&rnd);
    b[80..82].copy_from_slice(&f16::from_f32(0.0137).to_bits().to_le_bytes());
    b[82..84].copy_from_slice(&f16::from_f32(0.0091).to_bits().to_le_bytes());
    let buf = Aligned(b);

    let blocks = BlockQ2K::slice_from_bytes(&buf.0).expect("parse");
    let mut got = vec![0.0f32; QK_K];
    BlockQ2K::dequant(blocks, &mut got).expect("dequant");

    assert_bit_exact("Q2_K dense", &got, &ggml_ref::dequantize_row_q2_k(&buf.0));
}

// ===========================================================================
// Q3_K
// ===========================================================================

/// Hand-computed Q3_K golden.
///
/// `d = 0.25`; `scales[0] = 0x02` and `scales[8] = 0x02`, i.e. sub-block 0's
/// 6-bit code is `low = 2 | (high = 2) << 4 = 34`, so its signed scale is
/// `34 - 32 = +2`; every other sub-block's code is 0, i.e. `-32`.
/// `hmask[0] = 0x01` (the high bit of element 0 only), `qs[0] = 0x03`.
///
/// ggml subtracts 4 when the `hmask` bit is **clear** (`- (hm & m ? 0 : 4)`):
///
/// - `y[0]       = 0.25*(+2) * (3 - 0)  =  1.5`   (hmask bit set)
/// - `y[1..16]   = 0.25*(+2) * (0 - 4)  = -2.0`
/// - `y[16..256] = 0.25*(-32) * (0 - 4) = 32.0`   (all remaining codes are 0)
///
/// Element 32 lands in the `shift = 2`, `m = 2` iteration, so it reads
/// `hmask[0] & 2 == 0` and stays at 32.0 — pinning the `m <<= 1` stepping.
fn q3k_sparse_golden() -> Aligned<110> {
    let mut b = [0u8; 110];
    b[0] = 0x01; // hmask[0]
    b[32] = 0x03; // qs[0]
    b[96] = 0x02; // scales[0]  -> sub-block 0 low nibble = 2
    b[104] = 0x02; // scales[8] -> sub-block 0 high 2 bits = 2
    b[108..110].copy_from_slice(&f16::from_f32(0.25).to_bits().to_le_bytes());
    Aligned(b)
}

#[test]
fn q3k_hand_computed_golden_block() {
    let buf = q3k_sparse_golden();
    let blocks = BlockQ3K::slice_from_bytes(&buf.0).expect("aligned 110-byte block parses");
    assert_eq!(blocks.len(), 1);

    let mut got = vec![0.0f32; QK_K];
    BlockQ3K::dequant(blocks, &mut got).expect("dequant");

    let mut want = vec![32.0f32; QK_K];
    want[0] = 1.5;
    for w in want.iter_mut().take(16).skip(1) {
        *w = -2.0;
    }
    assert_exact("Q3_K hand golden", &got, &want);
    assert_bit_exact(
        "Q3_K hand golden vs ggml",
        &got,
        &ggml_ref::dequantize_row_q3_k(&buf.0),
    );
}

#[test]
fn q3k_dense_golden_matches_ggml_reference() {
    let rnd = lcg_bytes(108, 0x3BEE_0002);
    let mut b = [0u8; 110];
    b[0..108].copy_from_slice(&rnd);
    b[108..110].copy_from_slice(&f16::from_f32(0.0037).to_bits().to_le_bytes());
    let buf = Aligned(b);

    let blocks = BlockQ3K::slice_from_bytes(&buf.0).expect("parse");
    let mut got = vec![0.0f32; QK_K];
    BlockQ3K::dequant(blocks, &mut got).expect("dequant");

    assert_bit_exact("Q3_K dense", &got, &ggml_ref::dequantize_row_q3_k(&buf.0));
}

// ===========================================================================
// Q4_K
// ===========================================================================

/// Hand-computed Q4_K golden.
///
/// `d = 0.5`, `dmin = 0.25`, `qs[0] = 0x57` (low nibble 7, high nibble 5),
/// every other `qs` byte zero, and a `scales` array that exercises both halves
/// of `get_scale_min_k4`:
///
/// `scales[0] = 132`, `scales[1] = 3`, `scales[4] = 66`, `scales[5] = 1`,
/// `scales[8] = 0x31`.
///
/// ```text
/// j = 0 (j < 4):  sc = 132 & 63 = 4                        m = 66 & 63 = 2
/// j = 1        :  sc = 3                                   m = 1
/// j = 4 (j >= 4): sc = (0x31 & 0xF) | ((132 >> 6) << 4)    m = (0x31 >> 4) | ((66 >> 6) << 4)
///                    = 1 | 32 = 33                            = 3 | 16 = 19
/// every other sub-block resolves to sc = m = 0
/// ```
///
/// ggml emits 32 **low** nibbles then 32 **high** nibbles per 64-element group:
///
/// ```text
/// y[0]        = 0.5*4  * (0x57 & 0xF) - 0.25*2 = 2.0*7 - 0.5  = 13.5
/// y[1..32]    = 2.0*0 - 0.5                                   = -0.5
/// y[32]       = 0.5*3  * (0x57 >> 4)  - 0.25*1 = 1.5*5 - 0.25 =  7.25
/// y[33..64]   = 1.5*0 - 0.25                                  = -0.25
/// y[64..128]  = 0   (sub-blocks 2 and 3 are zero)
/// y[128..160] = 0.5*33 * 0 - 0.25*19                          = -4.75
/// y[160..256] = 0
/// ```
fn q4k_sparse_golden() -> Aligned<144> {
    let mut b = [0u8; 144];
    b[0..2].copy_from_slice(&f16::from_f32(0.5).to_bits().to_le_bytes());
    b[2..4].copy_from_slice(&f16::from_f32(0.25).to_bits().to_le_bytes());
    b[4] = 132; // scales[0]
    b[5] = 3; // scales[1]
    b[8] = 66; // scales[4]
    b[9] = 1; // scales[5]
    b[12] = 0x31; // scales[8]
    b[16] = 0x57; // qs[0]
    Aligned(b)
}

#[test]
fn q4k_hand_computed_golden_block() {
    let buf = q4k_sparse_golden();
    let blocks = BlockQ4K::slice_from_bytes(&buf.0).expect("aligned 144-byte block parses");
    assert_eq!(blocks.len(), 1);

    let mut got = vec![0.0f32; QK_K];
    BlockQ4K::dequant(blocks, &mut got).expect("dequant");

    let mut want = vec![0.0f32; QK_K];
    want[0] = 13.5;
    for w in want.iter_mut().take(32).skip(1) {
        *w = -0.5;
    }
    want[32] = 7.25;
    for w in want.iter_mut().take(64).skip(33) {
        *w = -0.25;
    }
    for w in want.iter_mut().take(160).skip(128) {
        *w = -4.75;
    }
    assert_exact("Q4_K hand golden", &got, &want);
    assert_bit_exact(
        "Q4_K hand golden vs ggml",
        &got,
        &ggml_ref::dequantize_row_q4_k(&buf.0),
    );
}

#[test]
fn q4k_dense_golden_matches_ggml_reference() {
    let rnd = lcg_bytes(140, 0x4C0D_0003);
    let mut b = [0u8; 144];
    b[0..2].copy_from_slice(&f16::from_f32(0.0211).to_bits().to_le_bytes());
    b[2..4].copy_from_slice(&f16::from_f32(0.0074).to_bits().to_le_bytes());
    b[4..144].copy_from_slice(&rnd);
    let buf = Aligned(b);

    let blocks = BlockQ4K::slice_from_bytes(&buf.0).expect("parse");
    let mut got = vec![0.0f32; QK_K];
    BlockQ4K::dequant(blocks, &mut got).expect("dequant");

    assert_bit_exact("Q4_K dense", &got, &ggml_ref::dequantize_row_q4_k(&buf.0));
}

// ===========================================================================
// Q5_K
// ===========================================================================

/// Hand-computed Q5_K golden.
///
/// Same scale layout idea as Q4_K, plus the 5th bit: `d = 0.5`, `dmin = 0.25`,
/// `scales[0] = 4`, `scales[1] = 3`, `scales[4] = 2`, `scales[5] = 1`,
/// `qh[0] = 0x03`, `qs[0] = 0x57`.
///
/// For the first 64-element group ggml uses `u1 = 1`, `u2 = 2`:
///
/// - `y[0]      = 0.5*4 * ((0x57 & 0xF) + 16) - 0.25*2 = 2.0*23 - 0.5  = 45.5`
/// - `y[1..32]  = 2.0*0 - 0.5                           = -0.5`
/// - `y[32]     = 0.5*3 * ((0x57 >> 4)  + 16) - 0.25*1 = 1.5*21 - 0.25 = 31.25`
/// - `y[33..64] = 1.5*0 - 0.25                          = -0.25`
/// - `y[64..256] = 0` (all remaining scales resolve to zero)
///
/// The `+16` on both lanes only appears if the decoder applies `u1`/`u2` (bit 0
/// and bit 1 of `qh[0]`) rather than a per-element `qh[i/8] >> (i%8)` bit.
fn q5k_sparse_golden() -> Aligned<176> {
    let mut b = [0u8; 176];
    b[0..2].copy_from_slice(&f16::from_f32(0.5).to_bits().to_le_bytes());
    b[2..4].copy_from_slice(&f16::from_f32(0.25).to_bits().to_le_bytes());
    b[4] = 4; // scales[0]
    b[5] = 3; // scales[1]
    b[8] = 2; // scales[4]
    b[9] = 1; // scales[5]
    b[16] = 0x03; // qh[0]
    b[48] = 0x57; // qs[0]
    Aligned(b)
}

#[test]
fn q5k_hand_computed_golden_block() {
    let buf = q5k_sparse_golden();
    let blocks = BlockQ5K::slice_from_bytes(&buf.0).expect("aligned 176-byte block parses");
    assert_eq!(blocks.len(), 1);

    let mut got = vec![0.0f32; QK_K];
    BlockQ5K::dequant(blocks, &mut got).expect("dequant");

    let mut want = vec![0.0f32; QK_K];
    want[0] = 45.5;
    for w in want.iter_mut().take(32).skip(1) {
        *w = -0.5;
    }
    want[32] = 31.25;
    for w in want.iter_mut().take(64).skip(33) {
        *w = -0.25;
    }
    assert_exact("Q5_K hand golden", &got, &want);
    assert_bit_exact(
        "Q5_K hand golden vs ggml",
        &got,
        &ggml_ref::dequantize_row_q5_k(&buf.0),
    );
}

#[test]
fn q5k_dense_golden_matches_ggml_reference() {
    let rnd = lcg_bytes(172, 0x5D1E_0004);
    let mut b = [0u8; 176];
    b[0..2].copy_from_slice(&f16::from_f32(0.0163).to_bits().to_le_bytes());
    b[2..4].copy_from_slice(&f16::from_f32(0.0052).to_bits().to_le_bytes());
    b[4..176].copy_from_slice(&rnd);
    let buf = Aligned(b);

    let blocks = BlockQ5K::slice_from_bytes(&buf.0).expect("parse");
    let mut got = vec![0.0f32; QK_K];
    BlockQ5K::dequant(blocks, &mut got).expect("dequant");

    assert_bit_exact("Q5_K dense", &got, &ggml_ref::dequantize_row_q5_k(&buf.0));
}

// ===========================================================================
// Q6_K
// ===========================================================================

/// Hand-computed Q6_K golden.
///
/// `d = 0.5`; `scales = [2, 0, 3, 0, -1, 0, 4, 0, 0, …]`;
/// `ql[0] = 0x9A`, `ql[32] = 0x27`, `qh[0] = 0xE4` (2-bit fields 0, 1, 2, 3).
///
/// ggml emits four interleaved lanes per `l`, with `is = l / 16`:
///
/// - `q1 = (0xA | (0 << 4)) - 32 = -22` -> `y[0]  = 0.5 * sc[0] * -22 = -22.0`
/// - `q2 = (0x7 | (1 << 4)) - 32 =  -9` -> `y[32] = 0.5 * sc[2] *  -9 = -13.5`
/// - `q3 = (0x9 | (2 << 4)) - 32 =   9` -> `y[64] = 0.5 * sc[4] *   9 =  -4.5`
/// - `q4 = (0x2 | (3 << 4)) - 32 =  18` -> `y[96] = 0.5 * sc[6] *  18 =  36.0`
///
/// For `l in 1..16` every nibble is zero, so `q = -32` and
/// `y[l] = -32.0`, `y[l+32] = -48.0`, `y[l+64] = 16.0`, `y[l+96] = -64.0`;
/// for `l in 16..32` `is = 1` and `sc[1] = sc[3] = sc[5] = sc[7] = 0`, so those
/// stay at 0. The second 128-element half uses `sc[8..16]`, all zero.
fn q6k_sparse_golden() -> Aligned<210> {
    let mut b = [0u8; 210];
    b[0] = 0x9A; // ql[0]
    b[32] = 0x27; // ql[32]
    b[128] = 0xE4; // qh[0]
    b[192] = 2i8 as u8; // scales[0]
    b[194] = 3i8 as u8; // scales[2]
    b[196] = (-1i8) as u8; // scales[4]
    b[198] = 4i8 as u8; // scales[6]
    b[208..210].copy_from_slice(&f16::from_f32(0.5).to_bits().to_le_bytes());
    Aligned(b)
}

#[test]
fn q6k_hand_computed_golden_block() {
    let buf = q6k_sparse_golden();
    let blocks = BlockQ6K::slice_from_bytes(&buf.0).expect("aligned 210-byte block parses");
    assert_eq!(blocks.len(), 1);

    let mut got = vec![0.0f32; QK_K];
    BlockQ6K::dequant(blocks, &mut got).expect("dequant");

    let mut want = vec![0.0f32; QK_K];
    want[0] = -22.0;
    want[32] = -13.5;
    want[64] = -4.5;
    want[96] = 36.0;
    for l in 1..16usize {
        want[l] = -32.0;
        want[l + 32] = -48.0;
        want[l + 64] = 16.0;
        want[l + 96] = -64.0;
    }
    assert_exact("Q6_K hand golden", &got, &want);
    assert_bit_exact(
        "Q6_K hand golden vs ggml",
        &got,
        &ggml_ref::dequantize_row_q6_k(&buf.0),
    );
}

#[test]
fn q6k_dense_golden_matches_ggml_reference() {
    let rnd = lcg_bytes(208, 0x6E2F_0005);
    let mut b = [0u8; 210];
    b[0..208].copy_from_slice(&rnd);
    b[208..210].copy_from_slice(&f16::from_f32(0.00042).to_bits().to_le_bytes());
    let buf = Aligned(b);

    let blocks = BlockQ6K::slice_from_bytes(&buf.0).expect("parse");
    let mut got = vec![0.0f32; QK_K];
    BlockQ6K::dequant(blocks, &mut got).expect("dequant");

    assert_bit_exact("Q6_K dense", &got, &ggml_ref::dequantize_row_q6_k(&buf.0));
}

// ===========================================================================
// Encoder -> ggml-decoder round trip
// ===========================================================================

/// Serialize the crate's blocks back to their on-disk bytes, field by field, so
/// the round-trip test also pins the struct layout.
fn q2k_bytes(b: &BlockQ2K) -> Vec<u8> {
    let mut out = Vec::with_capacity(84);
    out.extend_from_slice(&b.scales);
    out.extend_from_slice(&b.qs);
    out.extend_from_slice(&b.d.to_bits().to_le_bytes());
    out.extend_from_slice(&b.dmin.to_bits().to_le_bytes());
    out
}

fn q3k_bytes(b: &BlockQ3K) -> Vec<u8> {
    let mut out = Vec::with_capacity(110);
    out.extend_from_slice(&b.hmask);
    out.extend_from_slice(&b.qs);
    out.extend_from_slice(&b.scales);
    out.extend_from_slice(&b.d.to_bits().to_le_bytes());
    out
}

fn q4k_bytes(b: &BlockQ4K) -> Vec<u8> {
    let mut out = Vec::with_capacity(144);
    out.extend_from_slice(&b.d.to_bits().to_le_bytes());
    out.extend_from_slice(&b.dmin.to_bits().to_le_bytes());
    out.extend_from_slice(&b.scales);
    out.extend_from_slice(&b.qs);
    out
}

fn q5k_bytes(b: &BlockQ5K) -> Vec<u8> {
    let mut out = Vec::with_capacity(176);
    out.extend_from_slice(&b.d.to_bits().to_le_bytes());
    out.extend_from_slice(&b.dmin.to_bits().to_le_bytes());
    out.extend_from_slice(&b.scales);
    out.extend_from_slice(&b.qh);
    out.extend_from_slice(&b.qs);
    out
}

fn q6k_bytes(b: &BlockQ6K) -> Vec<u8> {
    let mut out = Vec::with_capacity(210);
    out.extend_from_slice(&b.ql);
    out.extend_from_slice(&b.qh);
    out.extend_from_slice(&b.scales.iter().map(|&v| v as u8).collect::<Vec<u8>>());
    out.extend_from_slice(&b.d.to_bits().to_le_bytes());
    out
}

/// The headline regression: the OxiBonsai encoder's output, decoded by the
/// **ggml** reference, must reconstruct the signal.
///
/// Before the rewrite this scored 1.20 (Q2_K) to 5.64 (Q3_K) max error because
/// the codecs were only self-consistent. ggml-exact codecs land inside the
/// format's own quantization step.
#[test]
fn kquant_encoder_roundtrips_through_ggml_decoder() {
    let x = smooth_signal(QK_K);

    // Guard against a degenerate (near-constant) signal making this vacuous.
    let span =
        x.iter().copied().fold(f32::MIN, f32::max) - x.iter().copied().fold(f32::MAX, f32::min);
    assert!(
        span > 1.5,
        "round-trip signal must span a real range, got {span}"
    );

    let q2 = BlockQ2K::quantize(&x).expect("q2_K quantize");
    let e2 = max_abs_err(&x, &ggml_ref::dequantize_row_q2_k(&q2k_bytes(&q2[0])));

    let q3 = BlockQ3K::quantize(&x).expect("q3_K quantize");
    let e3 = max_abs_err(&x, &ggml_ref::dequantize_row_q3_k(&q3k_bytes(&q3[0])));

    let q4 = BlockQ4K::quantize(&x).expect("q4_K quantize");
    let e4 = max_abs_err(&x, &ggml_ref::dequantize_row_q4_k(&q4k_bytes(&q4[0])));

    let q5 = BlockQ5K::quantize(&x).expect("q5_K quantize");
    let e5 = max_abs_err(&x, &ggml_ref::dequantize_row_q5_k(&q5k_bytes(&q5[0])));

    let q6 = BlockQ6K::quantize(&x).expect("q6_K quantize");
    let e6 = max_abs_err(&x, &ggml_ref::dequantize_row_q6_k(&q6k_bytes(&q6[0])));

    eprintln!(
        "encoder -> ggml decoder max error: Q2_K {e2:.4} Q3_K {e3:.4} \
         Q4_K {e4:.4} Q5_K {e5:.4} Q6_K {e6:.4}"
    );

    // Every format must land inside the 0.02-0.21 self-consistent band, i.e.
    // no worse than its own quantization step. The byte-incompatible codecs
    // scored 1.20 (Q2_K) to 5.64 (Q3_K) on this same assertion.
    for (label, err) in [
        ("Q2_K", e2),
        ("Q3_K", e3),
        ("Q4_K", e4),
        ("Q5_K", e5),
        ("Q6_K", e6),
    ] {
        assert!(
            err < 0.21,
            "{label} encoder -> ggml decoder max error {err} is outside the \
             0.02-0.21 self-consistent band (pre-fix it was 1.20-5.64)"
        );
    }

    // Bit depth must still buy accuracy: more bits, less error.
    assert!(e6 < e5, "Q6_K ({e6}) should beat Q5_K ({e5})");
    assert!(e5 < e4, "Q5_K ({e5}) should beat Q4_K ({e4})");
    assert!(e4 < e2, "Q4_K ({e4}) should beat Q2_K ({e2})");
}

/// The crate's own decoder must agree bit-for-bit with the ggml reference on
/// the crate's own encoder output — i.e. the encode and decode halves were
/// transliterated from the same layout.
#[test]
fn kquant_encoder_output_decodes_identically_in_both_decoders() {
    let x = smooth_signal(QK_K * 2);

    let q2 = BlockQ2K::quantize(&x).expect("quantize");
    let mut got = vec![0.0f32; QK_K * 2];
    BlockQ2K::dequant(&q2, &mut got).expect("dequant");
    for (i, blk) in q2.iter().enumerate() {
        let want = ggml_ref::dequantize_row_q2_k(&q2k_bytes(blk));
        assert_bit_exact("Q2_K encode/decode", &got[i * QK_K..(i + 1) * QK_K], &want);
    }

    let q3 = BlockQ3K::quantize(&x).expect("quantize");
    let mut got = vec![0.0f32; QK_K * 2];
    BlockQ3K::dequant(&q3, &mut got).expect("dequant");
    for (i, blk) in q3.iter().enumerate() {
        let want = ggml_ref::dequantize_row_q3_k(&q3k_bytes(blk));
        assert_bit_exact("Q3_K encode/decode", &got[i * QK_K..(i + 1) * QK_K], &want);
    }

    let q4 = BlockQ4K::quantize(&x).expect("quantize");
    let mut got = vec![0.0f32; QK_K * 2];
    BlockQ4K::dequant(&q4, &mut got).expect("dequant");
    for (i, blk) in q4.iter().enumerate() {
        let want = ggml_ref::dequantize_row_q4_k(&q4k_bytes(blk));
        assert_bit_exact("Q4_K encode/decode", &got[i * QK_K..(i + 1) * QK_K], &want);
    }

    let q5 = BlockQ5K::quantize(&x).expect("quantize");
    let mut got = vec![0.0f32; QK_K * 2];
    BlockQ5K::dequant(&q5, &mut got).expect("dequant");
    for (i, blk) in q5.iter().enumerate() {
        let want = ggml_ref::dequantize_row_q5_k(&q5k_bytes(blk));
        assert_bit_exact("Q5_K encode/decode", &got[i * QK_K..(i + 1) * QK_K], &want);
    }

    let q6 = BlockQ6K::quantize(&x).expect("quantize");
    let mut got = vec![0.0f32; QK_K * 2];
    BlockQ6K::dequant(&q6, &mut got).expect("dequant");
    for (i, blk) in q6.iter().enumerate() {
        let want = ggml_ref::dequantize_row_q6_k(&q6k_bytes(blk));
        assert_bit_exact("Q6_K encode/decode", &got[i * QK_K..(i + 1) * QK_K], &want);
    }
}

// ===========================================================================
// Q4_0 / Q8_0
// ===========================================================================

fn q4_0_bytes(b: &BlockQ4_0) -> Vec<u8> {
    let mut out = Vec::with_capacity(18);
    out.extend_from_slice(&b.d.to_bits().to_le_bytes());
    out.extend_from_slice(&b.qs);
    out
}

fn q8_0_bytes(b: &BlockQ8_0) -> Vec<u8> {
    let mut out = Vec::with_capacity(34);
    out.extend_from_slice(&b.d.to_bits().to_le_bytes());
    out.extend_from_slice(&b.qs.iter().map(|&v| v as u8).collect::<Vec<u8>>());
    out
}

/// Hand-computed Q4_0 block for a signal whose extremum is **negative**.
///
/// `x[0] = -0.7`, every other sample `0.0`:
///
/// - `amax = 0.7` but `max = -0.7` (the *signed* value at that position)
/// - `d  = max / -8 = +0.0875` — positive precisely because `max` is negative.
///   `0.0875 = 1.4 * 2^-4`; f16 keeps 10 mantissa bits, so it rounds to
///   `(1 + 410/1024) * 2^-4`, i.e. bits `(11 << 10) | 410 = 0x2D9A`.
/// - `id = 1/d = 11.428…` (from the **f32** `d`, as ggml does)
/// - `x[0]*id = -8` -> `(int8_t)(-8 + 8.5) = 0` -> nibble **0**
/// - every other sample -> `(int8_t)(0 + 8.5) = 8` -> nibble 8
///
/// so `qs[0] = 0 | (8 << 4) = 0x80` and `qs[1..16] = 0x88`.
///
/// The pre-fix encoder used `scale = max_abs / 7` (always non-negative) and
/// therefore could never emit nibble 0 — it would have produced `qs[0] = 0x81`
/// with `d = +0.1`, throwing away one of the sixteen levels.
#[test]
fn q4_0_negative_extremum_matches_hand_computed_ggml_block() {
    let mut x = [0.0f32; QK_Q4_0];
    x[0] = -0.7;

    let blocks = BlockQ4_0::quantize(&x).expect("quantize");
    assert_eq!(blocks.len(), 1);
    let got = q4_0_bytes(&blocks[0]);

    let mut want = Vec::with_capacity(18);
    want.extend_from_slice(&0x2D9Au16.to_le_bytes()); // d = +0.0875 in f16
    want.push(0x80); // nibble 0 (element 0) | nibble 8 (element 16)
    want.extend_from_slice(&[0x88u8; 15]);

    assert_eq!(
        got, want,
        "Q4_0 block bytes must match the hand-computed ggml block"
    );

    // Sanity: d is positive because the extremum was negative, and the most
    // negative sample round-trips exactly.
    assert!(
        blocks[0].d.to_f32() > 0.0,
        "d = max / -8 must be positive here"
    );
    let decoded = ggml_ref::dequantize_row_q4_0(&got);
    assert!(
        (decoded[0] - (-0.7)).abs() < 1e-3,
        "extremum should round-trip: got {}",
        decoded[0]
    );
}

/// The full `quantize_row_q4_0_ref` transliteration must agree byte for byte
/// with the crate encoder on a signal that exercises both signs of `d`.
#[test]
fn q4_0_encoder_is_byte_identical_to_ggml_reference() {
    // Two blocks: the first has a negative extremum, the second a positive one.
    let mut x = vec![0.0f32; QK_Q4_0 * 2];
    for (i, v) in x.iter_mut().enumerate() {
        let t = i as f32;
        *v = 0.9 * (t * 0.21).sin() - 0.35 * (t * 0.07).cos();
    }
    x[5] = -1.37; // negative extremum in block 0
    x[40] = 1.61; // positive extremum in block 1

    let blocks = BlockQ4_0::quantize(&x).expect("quantize");
    assert_eq!(blocks.len(), 2);

    let got: Vec<u8> = blocks.iter().flat_map(q4_0_bytes).collect();
    let want = ggml_ref::quantize_row_q4_0_ref(&x);
    assert_eq!(got, want, "Q4_0 encoder must be byte-identical to ggml");

    // Both signs of d must actually be present, otherwise the test is vacuous.
    assert!(blocks[0].d.to_f32() > 0.0, "block 0 d should be positive");
    assert!(blocks[1].d.to_f32() < 0.0, "block 1 d should be negative");
}

/// Q4_0 decode was already ggml-exact (0/32 mismatch) — keep it that way.
#[test]
fn q4_0_decoder_stays_zero_of_32_mismatch() {
    let rnd = lcg_bytes(16, 0x40F0_0006);
    let mut b = [0u8; 18];
    b[0..2].copy_from_slice(&f16::from_f32(-0.0731).to_bits().to_le_bytes());
    b[2..18].copy_from_slice(&rnd);
    let buf = Aligned(b);

    let blocks = BlockQ4_0::slice_from_bytes(&buf.0).expect("parse");
    let mut got = vec![0.0f32; QK_Q4_0];
    BlockQ4_0::dequant(blocks, &mut got).expect("dequant");

    assert_bit_exact("Q4_0 decode", &got, &ggml_ref::dequantize_row_q4_0(&buf.0));

    // The encoder's own output must decode identically too.
    let x = smooth_signal(QK_Q4_0);
    let enc = BlockQ4_0::quantize(&x).expect("quantize");
    let mut got = vec![0.0f32; QK_Q4_0];
    BlockQ4_0::dequant(&enc, &mut got).expect("dequant");
    assert_bit_exact(
        "Q4_0 encode/decode",
        &got,
        &ggml_ref::dequantize_row_q4_0(&q4_0_bytes(&enc[0])),
    );
}

/// Q8_0 decode was already ggml-exact (0/32 mismatch) — keep it that way.
#[test]
fn q8_0_decoder_stays_zero_of_32_mismatch() {
    let rnd = lcg_bytes(32, 0x80E0_0007);
    let mut b = [0u8; 34];
    b[0..2].copy_from_slice(&f16::from_f32(0.0119).to_bits().to_le_bytes());
    b[2..34].copy_from_slice(&rnd);
    let buf = Aligned(b);

    let blocks = BlockQ8_0::slice_from_bytes(&buf.0).expect("parse");
    let mut got = vec![0.0f32; QK_Q8_0];
    BlockQ8_0::dequant(blocks, &mut got).expect("dequant");

    assert_bit_exact("Q8_0 decode", &got, &ggml_ref::dequantize_row_q8_0(&buf.0));

    let x = smooth_signal(QK_Q8_0);
    let enc = BlockQ8_0::quantize(&x).expect("quantize");
    let mut got = vec![0.0f32; QK_Q8_0];
    BlockQ8_0::dequant(&enc, &mut got).expect("dequant");
    assert_bit_exact(
        "Q8_0 encode/decode",
        &got,
        &ggml_ref::dequantize_row_q8_0(&q8_0_bytes(&enc[0])),
    );
}

// ===========================================================================
// Primitive pins
// ===========================================================================

/// `nearest_int` rounds half-to-**even**, unlike `f32::round`. One differing
/// code on a `.5` boundary is enough to break bit-exactness against llama.cpp.
#[test]
fn nearest_int_rounds_half_to_even() {
    for (input, expected) in [
        (0.5f32, 0i32),
        (1.5, 2),
        (2.5, 2),
        (3.5, 4),
        (-0.5, 0),
        (-1.5, -2),
        (-2.5, -2),
        (0.49, 0),
        (0.51, 1),
        (-7.5, -8),
        (17.0, 17),
    ] {
        assert_eq!(
            ggml_ref::nearest_int(input),
            expected,
            "nearest_int({input}) should be {expected}"
        );
    }
    // f32::round() disagrees on exactly the half-way cases — that is the point.
    assert_ne!(ggml_ref::nearest_int(2.5), 2.5f32.round() as i32);
}

/// Block sizes must stay pinned to ggml's `static_assert`s.
#[test]
fn block_sizes_match_ggml() {
    assert_eq!(std::mem::size_of::<BlockQ2K>(), 84);
    assert_eq!(std::mem::size_of::<BlockQ3K>(), 110);
    assert_eq!(std::mem::size_of::<BlockQ4K>(), 144);
    assert_eq!(std::mem::size_of::<BlockQ5K>(), 176);
    assert_eq!(std::mem::size_of::<BlockQ6K>(), 210);
    assert_eq!(std::mem::size_of::<BlockQ4_0>(), 18);
    assert_eq!(std::mem::size_of::<BlockQ8_0>(), 34);
}
