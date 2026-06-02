//! A minimal, Pure-Rust 8-bit-RGB PNG encoder.
//!
//! This is the final stage of the Bonsai-Image text-to-image pipeline: it turns
//! the VAE decoder's RGB output (after the `clip(x/2 + 0.5)` → `u8` conversion)
//! into a standards-compliant PNG byte stream that any viewer can open.
//!
//! ## Format produced
//!
//! A baseline PNG (no interlacing, no ancillary chunks) consisting of exactly
//! three chunks after the 8-byte signature:
//!
//! 1. `IHDR` — width, height, `bit_depth = 8`, `color_type = 2` (truecolour
//!    RGB), compression `0` (DEFLATE), filter `0` (adaptive), interlace `0`.
//! 2. `IDAT` — the zlib-wrapped DEFLATE stream of the raw scanlines. Each
//!    scanline is a one-byte filter tag (`0` = `None`) followed by `width * 3`
//!    interleaved RGB bytes, exactly as the PNG spec mandates before
//!    compression.
//! 3. `IEND` — the empty end marker.
//!
//! Every chunk is framed as `len(u32 BE) ‖ type(4) ‖ data ‖ CRC32(u32 BE)`,
//! where the CRC is computed over `type ‖ data`.
//!
//! ## Dependencies (COOLJAPAN Pure Rust Policy)
//!
//! Compression uses [`oxiarc_deflate::zlib::zlib_compress`] (zlib-wrapped
//! DEFLATE — precisely PNG's `IDAT` encoding); the per-chunk CRC-32 uses
//! [`crc32fast`]. No C/C++/Fortran, no `zip`/`flate2`/`png`.
//!
//! ### Compression level
//!
//! We request DEFLATE level **9** (best compression). `oxiarc-deflate` 0.3.1
//! fixed the *dynamic*-Huffman incompleteness bug present in 0.3.0: previously
//! (at levels >= 5) the encoder emitted a code-length table its own inflater
//! accepted but that was not spec-compliant on larger inputs, so a conforming
//! decoder (zlib, libpng, browsers, Pillow) rejected it with "invalid code
//! lengths set". As of 0.3.1 every level 1–9 produces a spec-compliant stream
//! that round-trips through *any* conforming inflater, so the old level-4 cap is
//! no longer needed and we take the densest `IDAT` level 9 provides.
//!
//! ## Example
//!
//! ```
//! use oxibonsai_image::png::encode_rgb8;
//!
//! // A 2×1 image: one red pixel, one green pixel.
//! let rgb = [255u8, 0, 0, 0, 255, 0];
//! let png = encode_rgb8(2, 1, &rgb).expect("encode");
//! assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
//! ```

use oxiarc_deflate::zlib::zlib_compress;

/// The 8-byte PNG file signature.
const PNG_SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n'];

/// zlib/DEFLATE compression level used for the `IDAT` stream.
///
/// Set to 9 (best compression). The old cap at 4 worked around an
/// `oxiarc-deflate` 0.3.0 dynamic-Huffman bug that emitted non-spec-compliant
/// output at levels of 5 or more; `oxiarc-deflate` 0.3.1 fixed that, so every
/// level 1 through 9 now round-trips through any conforming inflater and we use
/// the densest. See the module docs.
const ZLIB_LEVEL: u8 = 9;

/// Errors that can occur while encoding a PNG.
#[derive(Debug, thiserror::Error)]
pub enum PngError {
    /// The supplied pixel buffer length did not equal `width * height * 3`.
    #[error("rgb buffer has {got} bytes, expected width*height*3 = {expected}")]
    BadLength {
        /// Number of bytes actually supplied.
        got: usize,
        /// Number of bytes required (`width * height * 3`).
        expected: usize,
    },

    /// `width` or `height` was zero (PNG forbids zero-dimension images).
    #[error("image dimensions must be non-zero (got {width}x{height})")]
    ZeroDimension {
        /// The supplied width.
        width: usize,
        /// The supplied height.
        height: usize,
    },

    /// A dimension exceeded the PNG limit (each is a 4-byte unsigned field).
    #[error("dimension {value} exceeds the PNG maximum of {}", u32::MAX)]
    DimensionTooLarge {
        /// The offending dimension value.
        value: usize,
    },

    /// The DEFLATE/zlib backend failed to compress the scanline buffer.
    #[error("zlib compression failed: {0}")]
    Compress(String),
}

/// Result alias for PNG encoding.
pub type PngResult<T> = Result<T, PngError>;

/// Encode an 8-bit RGB image into a complete PNG byte stream.
///
/// `rgb` must contain exactly `width * height * 3` bytes in row-major,
/// channel-interleaved (`R, G, B, R, G, B, …`) order — i.e. HWC layout.
///
/// # Errors
///
/// - [`PngError::ZeroDimension`] if `width` or `height` is `0`.
/// - [`PngError::DimensionTooLarge`] if either dimension exceeds [`u32::MAX`].
/// - [`PngError::BadLength`] if `rgb.len() != width * height * 3`.
/// - [`PngError::Compress`] if the zlib backend rejects the scanline buffer.
pub fn encode_rgb8(width: usize, height: usize, rgb: &[u8]) -> PngResult<Vec<u8>> {
    if width == 0 || height == 0 {
        return Err(PngError::ZeroDimension { width, height });
    }
    if width > u32::MAX as usize {
        return Err(PngError::DimensionTooLarge { value: width });
    }
    if height > u32::MAX as usize {
        return Err(PngError::DimensionTooLarge { value: height });
    }
    let row_bytes = width
        .checked_mul(3)
        .ok_or(PngError::DimensionTooLarge { value: width })?;
    let expected = row_bytes
        .checked_mul(height)
        .ok_or(PngError::DimensionTooLarge { value: height })?;
    if rgb.len() != expected {
        return Err(PngError::BadLength {
            got: rgb.len(),
            expected,
        });
    }

    // Build the raw, pre-compression scanline buffer: each row is a `0` filter
    // tag (filter type "None") followed by the row's RGB bytes.
    let mut raw = Vec::with_capacity(height * (1 + row_bytes));
    for row in 0..height {
        raw.push(0u8); // filter type 0 (None)
        let start = row * row_bytes;
        raw.extend_from_slice(&rgb[start..start + row_bytes]);
    }

    // IDAT payload = zlib(scanlines). zlib_compress emits the CMF/FLG header
    // (CM=8 DEFLATE, 32 KB window) + DEFLATE blocks + Adler-32 trailer, which is
    // exactly what PNG's IDAT requires.
    let idat = zlib_compress(&raw, ZLIB_LEVEL).map_err(|e| PngError::Compress(e.to_string()))?;

    // IHDR data: 13 bytes.
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&(width as u32).to_be_bytes());
    ihdr.extend_from_slice(&(height as u32).to_be_bytes());
    ihdr.push(8); // bit depth
    ihdr.push(2); // color type 2 = truecolour (RGB)
    ihdr.push(0); // compression method (DEFLATE)
    ihdr.push(0); // filter method (adaptive)
    ihdr.push(0); // interlace method (none)

    // Assemble the file: signature + IHDR + IDAT + IEND.
    let mut out = Vec::with_capacity(PNG_SIGNATURE.len() + 12 * 3 + ihdr.len() + idat.len());
    out.extend_from_slice(&PNG_SIGNATURE);
    write_chunk(&mut out, b"IHDR", &ihdr);
    write_chunk(&mut out, b"IDAT", &idat);
    write_chunk(&mut out, b"IEND", &[]);
    Ok(out)
}

/// Append one PNG chunk (`length ‖ type ‖ data ‖ CRC32`) to `out`.
///
/// All multi-byte integers are big-endian; the CRC-32 covers `chunk_type` and
/// `data` (but not the length field), per the PNG specification.
fn write_chunk(out: &mut Vec<u8>, chunk_type: &[u8; 4], data: &[u8]) {
    // Length is defined as the data length only; it always fits in u32 here
    // because IDAT is the only large chunk and is bounded by the image size,
    // which we have already validated against u32::MAX per dimension.
    let len = data.len() as u32;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(chunk_type);
    out.extend_from_slice(data);

    let mut hasher = crc32fast::Hasher::new();
    hasher.update(chunk_type);
    hasher.update(data);
    let crc = hasher.finalize();
    out.extend_from_slice(&crc.to_be_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiarc_deflate::zlib::zlib_decompress;

    /// Reconstruct the raw (filter-tagged) scanline buffer for a known image so
    /// the round-trip test can compare against the encoder's IDAT.
    fn raw_scanlines(width: usize, height: usize, rgb: &[u8]) -> Vec<u8> {
        let row_bytes = width * 3;
        let mut raw = Vec::with_capacity(height * (1 + row_bytes));
        for row in 0..height {
            raw.push(0u8);
            let start = row * row_bytes;
            raw.extend_from_slice(&rgb[start..start + row_bytes]);
        }
        raw
    }

    /// A minimal, self-contained, spec-compliant DEFLATE bit reader (LSB-first).
    struct BitReader<'a> {
        data: &'a [u8],
        byte: usize,
        bit: u32,
    }

    impl<'a> BitReader<'a> {
        fn new(data: &'a [u8]) -> Self {
            Self {
                data,
                byte: 0,
                bit: 0,
            }
        }

        /// Read a single bit (LSB-first within each byte).
        fn bit(&mut self) -> Result<u32, &'static str> {
            let b = *self.data.get(self.byte).ok_or("deflate: out of bits")?;
            let v = (b >> self.bit) & 1;
            self.bit += 1;
            if self.bit == 8 {
                self.bit = 0;
                self.byte += 1;
            }
            Ok(v as u32)
        }

        /// Read `n` bits LSB-first as a little-endian integer.
        fn bits(&mut self, n: u32) -> Result<u32, &'static str> {
            let mut v = 0u32;
            for i in 0..n {
                v |= self.bit()? << i;
            }
            Ok(v)
        }

        /// Align to the next byte boundary (for stored blocks).
        fn align(&mut self) {
            if self.bit != 0 {
                self.bit = 0;
                self.byte += 1;
            }
        }
    }

    /// A canonical Huffman decoder built from a code-length list.
    struct Huffman {
        /// `counts[len]` = number of codes of that bit-length.
        counts: [u16; 16],
        /// Symbols sorted by (length, symbol).
        symbols: Vec<u16>,
    }

    impl Huffman {
        fn new(lengths: &[u8]) -> Self {
            let mut counts = [0u16; 16];
            for &l in lengths {
                counts[l as usize] += 1;
            }
            counts[0] = 0;
            let mut offsets = [0u16; 16];
            let mut sum = 0u16;
            for len in 1..16 {
                offsets[len] = sum;
                sum += counts[len];
            }
            let mut symbols = vec![0u16; lengths.len()];
            for (sym, &l) in lengths.iter().enumerate() {
                if l != 0 {
                    symbols[offsets[l as usize] as usize] = sym as u16;
                    offsets[l as usize] += 1;
                }
            }
            Self { counts, symbols }
        }

        /// Decode one symbol from the bit stream (canonical Huffman).
        fn decode(&self, br: &mut BitReader) -> Result<u16, &'static str> {
            let mut code = 0i32;
            let mut first = 0i32;
            let mut index = 0i32;
            for len in 1..16 {
                code |= br.bit()? as i32;
                let count = self.counts[len] as i32;
                if code - first < count {
                    return Ok(self.symbols[(index + (code - first)) as usize]);
                }
                index += count;
                first += count;
                first <<= 1;
                code <<= 1;
            }
            Err("deflate: bad symbol")
        }
    }

    /// Independent, spec-compliant raw-DEFLATE inflater (stored + fixed +
    /// dynamic Huffman). Deliberately does NOT use `oxiarc-deflate`, so it
    /// validates that the produced stream is decodable by a conforming decoder.
    fn inflate_raw(data: &[u8]) -> Result<Vec<u8>, &'static str> {
        const LEN_BASE: [u16; 29] = [
            3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99,
            115, 131, 163, 195, 227, 258,
        ];
        const LEN_EXTRA: [u8; 29] = [
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
        ];
        const DIST_BASE: [u16; 30] = [
            1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025,
            1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
        ];
        const DIST_EXTRA: [u8; 30] = [
            0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12,
            12, 13, 13,
        ];
        const CL_ORDER: [usize; 19] = [
            16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
        ];

        let mut br = BitReader::new(data);
        let mut out = Vec::new();
        loop {
            let final_block = br.bit()?;
            let btype = br.bits(2)?;
            match btype {
                0 => {
                    br.align();
                    let len = u16::from_le_bytes([
                        *data.get(br.byte).ok_or("stored len")?,
                        *data.get(br.byte + 1).ok_or("stored len")?,
                    ]) as usize;
                    br.byte += 4; // skip LEN + NLEN
                    let chunk = data.get(br.byte..br.byte + len).ok_or("stored payload")?;
                    out.extend_from_slice(chunk);
                    br.byte += len;
                }
                1 | 2 => {
                    let (lit, dist) = if btype == 1 {
                        let mut ll = [0u8; 288];
                        for (i, slot) in ll.iter_mut().enumerate() {
                            *slot = if i < 144 {
                                8
                            } else if i < 256 {
                                9
                            } else if i < 280 {
                                7
                            } else {
                                8
                            };
                        }
                        let dd = [5u8; 30];
                        (Huffman::new(&ll), Huffman::new(&dd))
                    } else {
                        let hlit = br.bits(5)? as usize + 257;
                        let hdist = br.bits(5)? as usize + 1;
                        let hclen = br.bits(4)? as usize + 4;
                        let mut cl_lengths = [0u8; 19];
                        for &idx in CL_ORDER.iter().take(hclen) {
                            cl_lengths[idx] = br.bits(3)? as u8;
                        }
                        let cl = Huffman::new(&cl_lengths);
                        let mut lengths = Vec::with_capacity(hlit + hdist);
                        while lengths.len() < hlit + hdist {
                            let sym = cl.decode(&mut br)?;
                            match sym {
                                0..=15 => lengths.push(sym as u8),
                                16 => {
                                    let prev = *lengths.last().ok_or("repeat with no prev")?;
                                    let n = br.bits(2)? + 3;
                                    for _ in 0..n {
                                        lengths.push(prev);
                                    }
                                }
                                17 => {
                                    let n = br.bits(3)? as usize + 3;
                                    lengths.resize(lengths.len() + n, 0);
                                }
                                18 => {
                                    let n = br.bits(7)? as usize + 11;
                                    lengths.resize(lengths.len() + n, 0);
                                }
                                _ => return Err("bad cl symbol"),
                            }
                        }
                        if lengths.len() != hlit + hdist {
                            return Err("length set overrun");
                        }
                        let (ll, dd) = lengths.split_at(hlit);
                        (Huffman::new(ll), Huffman::new(dd))
                    };
                    loop {
                        let sym = lit.decode(&mut br)?;
                        if sym == 256 {
                            break;
                        } else if sym < 256 {
                            out.push(sym as u8);
                        } else {
                            let li = (sym - 257) as usize;
                            if li >= LEN_BASE.len() {
                                return Err("bad length symbol");
                            }
                            let length =
                                LEN_BASE[li] as usize + br.bits(LEN_EXTRA[li] as u32)? as usize;
                            let dsym = dist.decode(&mut br)? as usize;
                            if dsym >= DIST_BASE.len() {
                                return Err("bad dist symbol");
                            }
                            let distance = DIST_BASE[dsym] as usize
                                + br.bits(DIST_EXTRA[dsym] as u32)? as usize;
                            if distance == 0 || distance > out.len() {
                                return Err("bad back-reference");
                            }
                            let start = out.len() - distance;
                            for k in 0..length {
                                out.push(out[start + k]);
                            }
                        }
                    }
                }
                _ => return Err("reserved block type"),
            }
            if final_block == 1 {
                break;
            }
        }
        Ok(out)
    }

    /// Strip a 2-byte zlib header + 4-byte Adler-32 trailer and inflate the raw
    /// DEFLATE body with the independent inflater above.
    fn zlib_inflate_independent(data: &[u8]) -> Result<Vec<u8>, &'static str> {
        if data.len() < 6 {
            return Err("zlib too short");
        }
        // CMF/FLG sanity: CM must be 8 (DEFLATE).
        if data[0] & 0x0f != 8 {
            return Err("zlib: not DEFLATE");
        }
        inflate_raw(&data[2..data.len() - 4])
    }

    /// Locate the first chunk of `chunk_type` and return its data slice.
    fn find_chunk<'a>(png: &'a [u8], chunk_type: &[u8; 4]) -> Option<&'a [u8]> {
        let mut pos = 8usize; // skip signature
        while pos + 8 <= png.len() {
            let len =
                u32::from_be_bytes([png[pos], png[pos + 1], png[pos + 2], png[pos + 3]]) as usize;
            let ty = &png[pos + 4..pos + 8];
            let data_start = pos + 8;
            let data_end = data_start + len;
            if data_end + 4 > png.len() {
                return None;
            }
            if ty == chunk_type {
                return Some(&png[data_start..data_end]);
            }
            pos = data_end + 4; // skip data + CRC
        }
        None
    }

    #[test]
    fn signature_and_structure() {
        // 3×2 image with distinct pixels.
        let width = 3;
        let height = 2;
        let rgb: Vec<u8> = vec![
            0, 0, 0, 255, 255, 255, 255, 0, 0, // row 0: black, white, red
            0, 255, 0, 0, 0, 255, 128, 64, 32, // row 1: green, blue, brownish
        ];
        let png = encode_rgb8(width, height, &rgb).expect("encode");

        // Signature.
        assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");

        // IHDR present, 13 bytes, correct dims and color type.
        let ihdr = find_chunk(&png, b"IHDR").expect("IHDR");
        assert_eq!(ihdr.len(), 13);
        assert_eq!(
            u32::from_be_bytes([ihdr[0], ihdr[1], ihdr[2], ihdr[3]]),
            width as u32
        );
        assert_eq!(
            u32::from_be_bytes([ihdr[4], ihdr[5], ihdr[6], ihdr[7]]),
            height as u32
        );
        assert_eq!(ihdr[8], 8, "bit depth");
        assert_eq!(ihdr[9], 2, "color type RGB");
        assert_eq!(ihdr[10], 0, "compression");
        assert_eq!(ihdr[11], 0, "filter");
        assert_eq!(ihdr[12], 0, "interlace");

        // IEND present and empty.
        let iend = find_chunk(&png, b"IEND").expect("IEND");
        assert!(iend.is_empty());
    }

    #[test]
    fn idat_roundtrips_to_scanlines() {
        let width = 4;
        let height = 3;
        // Deterministic gradient.
        let mut rgb = Vec::with_capacity(width * height * 3);
        for y in 0..height {
            for x in 0..width {
                rgb.push((x * 60) as u8);
                rgb.push((y * 80) as u8);
                rgb.push(((x + y) * 30) as u8);
            }
        }
        let png = encode_rgb8(width, height, &rgb).expect("encode");

        let idat = find_chunk(&png, b"IDAT").expect("IDAT");
        let inflated = zlib_decompress(idat).expect("inflate IDAT");

        let expected = raw_scanlines(width, height, &rgb);
        assert_eq!(
            inflated, expected,
            "re-inflated IDAT must match the raw scanlines"
        );

        // And the recovered pixels (stripping the per-row filter byte) match the
        // original RGB buffer.
        let row_bytes = width * 3;
        let mut recovered = Vec::with_capacity(width * height * 3);
        for row in 0..height {
            let start = row * (1 + row_bytes);
            assert_eq!(inflated[start], 0, "filter byte must be 0 (None)");
            recovered.extend_from_slice(&inflated[start + 1..start + 1 + row_bytes]);
        }
        assert_eq!(recovered, rgb);
    }

    #[test]
    fn idat_decodes_with_independent_inflater() {
        // A larger, less-compressible image so the encoder exercises a real
        // (non-stored) DEFLATE block. At level 9 this engages the dynamic-Huffman
        // path that oxiarc-deflate 0.3.0 mis-encoded; oxiarc-deflate 0.3.1 fixed
        // it. This guards against emitting a stream that only oxiarc-deflate's own
        // inflater can read: it must decode with a fully independent,
        // spec-compliant inflater too.
        let width = 96;
        let height = 64;
        let mut rgb = Vec::with_capacity(width * height * 3);
        // Pseudo-random-ish but deterministic content (defeats trivial RLE).
        let mut state = 0x1234_5678u32;
        for _ in 0..width * height {
            for _ in 0..3 {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                rgb.push((state >> 24) as u8);
            }
        }
        let png = encode_rgb8(width, height, &rgb).expect("encode");
        let idat = find_chunk(&png, b"IDAT").expect("IDAT");

        let inflated =
            zlib_inflate_independent(idat).expect("IDAT must decode with an independent inflater");
        let expected = raw_scanlines(width, height, &rgb);
        assert_eq!(
            inflated, expected,
            "independent inflate of IDAT must match the raw scanlines"
        );
    }

    #[test]
    fn crc_is_correct_for_iend() {
        // IEND with empty data has a well-known CRC-32 of 0xAE426082.
        let png = encode_rgb8(1, 1, &[10, 20, 30]).expect("encode");
        // Find IEND framing: it is the last 12 bytes (len=0, "IEND", crc).
        let n = png.len();
        let crc = u32::from_be_bytes([png[n - 4], png[n - 3], png[n - 2], png[n - 1]]);
        assert_eq!(crc, 0xAE42_6082, "IEND CRC");
    }

    #[test]
    fn rejects_bad_length() {
        let err = encode_rgb8(2, 2, &[0, 0, 0]).expect_err("should reject short buffer");
        match err {
            PngError::BadLength { got, expected } => {
                assert_eq!(got, 3);
                assert_eq!(expected, 12);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn rejects_zero_dimension() {
        let err = encode_rgb8(0, 5, &[]).expect_err("should reject zero width");
        assert!(matches!(err, PngError::ZeroDimension { .. }));
    }
}
