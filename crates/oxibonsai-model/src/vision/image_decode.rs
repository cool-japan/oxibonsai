//! Turning an image reference into [`ImageRgb8`] pixels for the vision
//! tower: `data:` URIs, local files, PNG and JPEG decoding (design §6.2).
//!
//! # Matching the reference
//!
//! The PrismML fork's `libmtmd` reads images with `stb_image`
//! (`stbi_load_from_memory(.., 3)`), which composites nothing: an alpha
//! channel is simply dropped, grey is replicated into R, G and B, a palette
//! is expanded, and 16-bit samples keep their high byte. [`decode_image`]
//! produces the same RGB bytes:
//!
//! * **PNG** is decoded in-house over `oxiarc-deflate`: every colour type
//!   (grey, RGB, palette, grey + alpha, RGBA), every legal bit depth
//!   (1/2/4/8/16), Adam7 interlacing, all five scanline filters; sub-byte
//!   grey scales exactly like `stb_image` (`x 0xff / 0x55 / 0x11`). Critical
//!   chunk CRCs are verified; ancillary chunks are skipped.
//! * **JPEG** — baseline and progressive — goes through `jpeg-decoder`,
//!   whose IDCT and chroma upsampling derive from `stb_image`'s; its colour
//!   conversion rounds in a finer fixed point, so a JPEG can differ from the
//!   reference by one unit in a channel. CMYK / YCCK (Adobe) JPEGs are
//!   converted with `stb_image`'s `blinn_8x8` formula; 12/16-bit JPEG is
//!   refused, as the reference refuses it.
//!
//! # Limits
//!
//! Every input is caller data, so nothing here panics on it and every
//! allocation is bounded before it is made: an encoded image is refused
//! above [`MAX_ENCODED_IMAGE_BYTES`], a decoded one above
//! [`MAX_DECODED_PIXELS`] (checked from the header, before any pixel
//! buffer exists), and a PNG's inflated data must be exactly the size its
//! header declares.
//!
//! # Sources
//!
//! [`load_image_source`] resolves an OpenAI `image_url` / CLI `--image`
//! reference under an [`ImageSourcePolicy`]: base64 `data:` URIs always,
//! local files only where the policy allows them (the CLI names its own
//! files; a server resolves `file://` references inside an operator-chosen
//! media directory only — `OXI_MEDIA_PATH` for `oxibonsai serve`, as the
//! reference server's `--media-path` does), and
//! `http(s)` never fetched — fetching arbitrary URLs from a server is a
//! server-side request forgery surface, so it is refused with a typed error
//! whether or not the operator opted in, until a fetcher with an address
//! policy exists.

use std::path::{Component, Path, PathBuf};

use super::ImageRgb8;

/// The largest encoded image (file, data URI payload) accepted: 32 MiB.
pub const MAX_ENCODED_IMAGE_BYTES: usize = 32 * 1024 * 1024;

/// The largest decoded image accepted, in pixels (8192 x 8192).
pub const MAX_DECODED_PIXELS: usize = 8192 * 8192;

/// The largest side accepted, in pixels.
pub const MAX_IMAGE_SIDE: usize = 32_768;

/// Why an image reference could not become pixels (or, from
/// [`super::preprocess`], a tower-ready image). Every variant is a property
/// of the request, with a stable [`ImageInputError::code`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ImageInputError {
    /// The bytes are neither PNG nor JPEG.
    #[error("unsupported image format ({detected}); PNG and JPEG are accepted")]
    UnsupportedFormat {
        /// What the leading bytes look like.
        detected: String,
    },
    /// The image is PNG or JPEG but malformed.
    #[error("malformed {format} image: {reason}")]
    Malformed {
        /// `"PNG"` or `"JPEG"`.
        format: &'static str,
        /// What is wrong.
        reason: String,
    },
    /// A well-formed image using a feature this decoder (and the
    /// reference) does not support.
    #[error("unsupported {format} image: {reason}")]
    Unsupported {
        /// `"PNG"` or `"JPEG"`.
        format: &'static str,
        /// The feature.
        reason: String,
    },
    /// The encoded payload is larger than [`MAX_ENCODED_IMAGE_BYTES`].
    #[error("the encoded image is {bytes} bytes; at most {limit} are accepted")]
    EncodedTooLarge {
        /// Encoded size.
        bytes: usize,
        /// The limit.
        limit: usize,
    },
    /// The decoded image would exceed [`MAX_DECODED_PIXELS`] or
    /// [`MAX_IMAGE_SIDE`].
    #[error(
        "the image is {width} x {height} pixels; at most {max_pixels} pixels and {max_side} per \
         side are accepted"
    )]
    TooLarge {
        /// Declared width.
        width: usize,
        /// Declared height.
        height: usize,
        /// [`MAX_DECODED_PIXELS`].
        max_pixels: usize,
        /// [`MAX_IMAGE_SIDE`].
        max_side: usize,
    },
    /// The image has a zero side.
    #[error("the image is empty ({width} x {height})")]
    Empty {
        /// Width.
        width: usize,
        /// Height.
        height: usize,
    },
    /// Even the smallest grid the aspect-preserving resize allows exceeds
    /// the per-image token budget (`--image-max-tokens`).
    #[error(
        "a {width} x {height} image needs a {grid_h} x {grid_w} merged grid ({tokens} image \
         tokens) even at its smallest aspect-preserving size, over the budget of {max_tokens} \
         tokens per image (--image-max-tokens)"
    )]
    TooManyTokens {
        /// Source width.
        width: usize,
        /// Source height.
        height: usize,
        /// Merged-grid rows after resizing.
        grid_h: usize,
        /// Merged-grid columns after resizing.
        grid_w: usize,
        /// `grid_h * grid_w`.
        tokens: usize,
        /// The budget.
        max_tokens: usize,
    },
    /// A malformed `data:` URI or base64 payload.
    #[error("invalid data URI: {reason}")]
    DataUri {
        /// What is wrong.
        reason: String,
    },
    /// A reference scheme this build does not resolve.
    #[error("unsupported image reference: {reason}")]
    UnsupportedSource {
        /// What was given.
        reason: String,
    },
    /// An `http(s)` reference (never fetched; see the module docs).
    #[error(
        "remote image URLs are not fetched ({url}): {reason}; send the image inline as a base64 \
         data URI (data:image/png;base64,...)"
    )]
    RemoteFetchRefused {
        /// The URL, truncated for the message.
        url: String,
        /// Why (disabled, or enabled but unavailable).
        reason: String,
    },
    /// A local file reference the policy does not allow.
    #[error("local image file refused: {reason}")]
    LocalFileRefused {
        /// Why.
        reason: String,
    },
    /// A local file that cannot be read.
    #[error("cannot read image file {path}: {reason}")]
    FileUnreadable {
        /// The path as given.
        path: String,
        /// The I/O error.
        reason: String,
    },
    /// An invalid preprocessing configuration (e.g. a zero token budget).
    #[error("invalid image preprocessing configuration: {reason}")]
    InvalidConfig {
        /// What is wrong.
        reason: String,
    },
}

impl ImageInputError {
    /// A short, stable code for monitoring and API error bodies.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::UnsupportedFormat { .. } | Self::Unsupported { .. } => "image_format_unsupported",
            Self::Malformed { .. } => "image_decode_failed",
            Self::EncodedTooLarge { .. } | Self::TooLarge { .. } => "image_too_large",
            Self::Empty { .. } => "image_empty",
            Self::TooManyTokens { .. } => "image_too_many_tokens",
            Self::DataUri { .. } => "image_data_uri_invalid",
            Self::UnsupportedSource { .. } => "image_url_scheme_unsupported",
            Self::RemoteFetchRefused { .. } => "image_url_fetch_disabled",
            Self::LocalFileRefused { .. } => "image_file_refused",
            Self::FileUnreadable { .. } => "image_file_unreadable",
            Self::InvalidConfig { .. } => "image_config_invalid",
        }
    }
}

fn malformed_png(reason: impl Into<String>) -> ImageInputError {
    ImageInputError::Malformed {
        format: "PNG",
        reason: reason.into(),
    }
}

/// Refuse a declared size before any pixel buffer is allocated.
fn check_dimensions(width: usize, height: usize) -> Result<(), ImageInputError> {
    if width == 0 || height == 0 {
        return Err(ImageInputError::Empty { width, height });
    }
    let too_large = || ImageInputError::TooLarge {
        width,
        height,
        max_pixels: MAX_DECODED_PIXELS,
        max_side: MAX_IMAGE_SIDE,
    };
    if width > MAX_IMAGE_SIDE || height > MAX_IMAGE_SIDE {
        return Err(too_large());
    }
    match width.checked_mul(height) {
        Some(pixels) if pixels <= MAX_DECODED_PIXELS => Ok(()),
        _ => Err(too_large()),
    }
}

/// The encoded formats [`decode_image`] recognises.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    /// PNG (`\x89PNG\r\n\x1a\n`).
    Png,
    /// JPEG (`FF D8 FF`).
    Jpeg,
}

const PNG_SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n'];

/// Recognise PNG or JPEG from the leading bytes.
#[must_use]
pub fn sniff_format(bytes: &[u8]) -> Option<ImageFormat> {
    if bytes.starts_with(&PNG_SIGNATURE) {
        Some(ImageFormat::Png)
    } else if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        Some(ImageFormat::Jpeg)
    } else {
        None
    }
}

/// A name for a format this decoder refuses, for the error message.
fn describe_unknown(bytes: &[u8]) -> String {
    let named = if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        Some("GIF")
    } else if bytes.starts_with(b"BM") {
        Some("BMP")
    } else if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP" {
        Some("WebP")
    } else if bytes.starts_with(b"II*\0") || bytes.starts_with(b"MM\0*") {
        Some("TIFF")
    } else if bytes.len() >= 12 && &bytes[4..8] == b"ftyp" {
        Some("HEIF/AVIF")
    } else {
        None
    };
    match named {
        Some(name) => name.to_string(),
        None if bytes.is_empty() => "empty input".to_string(),
        None => {
            let head: Vec<String> = bytes.iter().take(8).map(|b| format!("{b:02x}")).collect();
            format!("unrecognised leading bytes {}", head.join(" "))
        }
    }
}

/// Decode a PNG or JPEG into RGB8 pixels the way the reference's
/// `stb_image` does (see the module docs).
///
/// # Errors
///
/// [`ImageInputError`] naming the format and the reason; never a panic.
pub fn decode_image(bytes: &[u8]) -> Result<ImageRgb8, ImageInputError> {
    if bytes.len() > MAX_ENCODED_IMAGE_BYTES {
        return Err(ImageInputError::EncodedTooLarge {
            bytes: bytes.len(),
            limit: MAX_ENCODED_IMAGE_BYTES,
        });
    }
    match sniff_format(bytes) {
        Some(ImageFormat::Png) => decode_png(bytes),
        Some(ImageFormat::Jpeg) => decode_jpeg(bytes),
        None => Err(ImageInputError::UnsupportedFormat {
            detected: describe_unknown(bytes),
        }),
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  PNG
// ─────────────────────────────────────────────────────────────────────────

/// CRC-32 (IEEE, reflected) table, built at compile time.
const CRC_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut n = 0;
    while n < 256 {
        let mut c = n as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
            k += 1;
        }
        table[n] = c;
        n += 1;
    }
    table
};

/// CRC-32 over `parts`, concatenated (a PNG chunk's CRC covers its type
/// and data).
fn crc32(parts: &[&[u8]]) -> u32 {
    let mut c = 0xFFFF_FFFFu32;
    for part in parts {
        for &b in *part {
            c = CRC_TABLE[((c ^ u32::from(b)) & 0xFF) as usize] ^ (c >> 8);
        }
    }
    c ^ 0xFFFF_FFFF
}

/// `IHDR`, validated.
#[derive(Debug, Clone, Copy)]
struct PngHeader {
    width: usize,
    height: usize,
    bit_depth: u8,
    color_type: u8,
    interlaced: bool,
}

impl PngHeader {
    fn parse(data: &[u8]) -> Result<Self, ImageInputError> {
        let [w0, w1, w2, w3, h0, h1, h2, h3, bit_depth, color_type, compression, filter, interlace] =
            <[u8; 13]>::try_from(data)
                .map_err(|_| malformed_png(format!("IHDR is {} bytes, not 13", data.len())))?;
        let width = u32::from_be_bytes([w0, w1, w2, w3]) as usize;
        let height = u32::from_be_bytes([h0, h1, h2, h3]) as usize;
        let depth_ok = match color_type {
            0 => matches!(bit_depth, 1 | 2 | 4 | 8 | 16),
            3 => matches!(bit_depth, 1 | 2 | 4 | 8),
            2 | 4 | 6 => matches!(bit_depth, 8 | 16),
            _ => return Err(malformed_png(format!("unknown colour type {color_type}"))),
        };
        if !depth_ok {
            return Err(malformed_png(format!(
                "bit depth {bit_depth} is not legal for colour type {color_type}"
            )));
        }
        if compression != 0 || filter != 0 {
            return Err(malformed_png(format!(
                "compression method {compression} / filter method {filter} (only 0 exists)"
            )));
        }
        if interlace > 1 {
            return Err(malformed_png(format!(
                "unknown interlace method {interlace}"
            )));
        }
        check_dimensions(width, height)?;
        Ok(Self {
            width,
            height,
            bit_depth,
            color_type,
            interlaced: interlace == 1,
        })
    }

    /// Samples per pixel.
    fn channels(&self) -> usize {
        match self.color_type {
            2 => 3,
            4 => 2,
            6 => 4,
            _ => 1,
        }
    }

    /// Bits per pixel.
    fn pixel_bits(&self) -> usize {
        self.channels() * usize::from(self.bit_depth)
    }

    /// Bytes of one filtered scanline of `width` pixels (without the
    /// filter-type byte).
    fn row_bytes(&self, width: usize) -> usize {
        (width * self.pixel_bits()).div_ceil(8)
    }

    /// The filter's "bytes per complete pixel" (at least 1).
    fn filter_bpp(&self) -> usize {
        self.pixel_bits().div_ceil(8).max(1)
    }
}

/// Adam7 passes: `(x0, y0, dx, dy)`.
const ADAM7: [(usize, usize, usize, usize); 7] = [
    (0, 0, 8, 8),
    (4, 0, 8, 8),
    (0, 4, 4, 8),
    (2, 0, 4, 4),
    (0, 2, 2, 4),
    (1, 0, 2, 2),
    (0, 1, 1, 2),
];

/// The sub-image dimensions of one Adam7 pass.
fn pass_dims(width: usize, height: usize, pass: (usize, usize, usize, usize)) -> (usize, usize) {
    let (x0, y0, dx, dy) = pass;
    let w = if width > x0 {
        (width - x0).div_ceil(dx)
    } else {
        0
    };
    let h = if height > y0 {
        (height - y0).div_ceil(dy)
    } else {
        0
    };
    (w, h)
}

fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let p = i16::from(a) + i16::from(b) - i16::from(c);
    let pa = (p - i16::from(a)).abs();
    let pb = (p - i16::from(b)).abs();
    let pc = (p - i16::from(c)).abs();
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

/// Undo the PNG filters of one (sub-)image in place: `data` is `height`
/// rows of `1 + row_bytes`; returns the unfiltered rows, concatenated.
fn unfilter(
    data: &[u8],
    height: usize,
    row_bytes: usize,
    bpp: usize,
) -> Result<Vec<u8>, ImageInputError> {
    let mut out = vec![0u8; height * row_bytes];
    for y in 0..height {
        let src = data
            .get(y * (row_bytes + 1)..(y + 1) * (row_bytes + 1))
            .ok_or_else(|| malformed_png("image data ends mid-scanline"))?;
        let (filter, line) = src
            .split_first()
            .ok_or_else(|| malformed_png("empty scanline"))?;
        let (done, rest) = out.split_at_mut(y * row_bytes);
        let prev: &[u8] = if y == 0 {
            &[]
        } else {
            &done[(y - 1) * row_bytes..]
        };
        let cur = &mut rest[..row_bytes];
        let up = |i: usize| prev.get(i).copied().unwrap_or(0);
        for i in 0..row_bytes {
            let x = line[i];
            let a = if i >= bpp { cur[i - bpp] } else { 0 };
            let b = up(i);
            let c = if i >= bpp { up(i - bpp) } else { 0 };
            cur[i] = match filter {
                0 => x,
                1 => x.wrapping_add(a),
                2 => x.wrapping_add(b),
                3 => x.wrapping_add(((u16::from(a) + u16::from(b)) / 2) as u8),
                4 => x.wrapping_add(paeth(a, b, c)),
                other => return Err(malformed_png(format!("unknown scanline filter {other}"))),
            };
        }
    }
    Ok(out)
}

/// `stb_image`'s sub-byte grey scale (`stbi__depth_scale_table`).
fn grey_scale(bit_depth: u8) -> u8 {
    match bit_depth {
        1 => 0xFF,
        2 => 0x55,
        4 => 0x11,
        _ => 1,
    }
}

/// Sample `index` of an unfiltered row at `bit_depth` bits (MSB first
/// below 8 bits, the high byte of a big-endian 16-bit sample).
fn sample(row: &[u8], index: usize, bit_depth: u8) -> u8 {
    match bit_depth {
        8 => row.get(index).copied().unwrap_or(0),
        16 => row.get(index * 2).copied().unwrap_or(0),
        bits => {
            let bits = usize::from(bits);
            let bit = index * bits;
            let byte = row.get(bit / 8).copied().unwrap_or(0);
            let shift = 8 - bits - (bit % 8);
            (byte >> shift) & ((1u8 << bits) - 1)
        }
    }
}

/// Convert one unfiltered row of `width` pixels to RGB8, like `stb_image`
/// with three requested channels.
fn row_to_rgb(
    header: &PngHeader,
    palette: &[[u8; 3]],
    row: &[u8],
    width: usize,
    out: &mut [u8],
) -> Result<(), ImageInputError> {
    let depth = header.bit_depth;
    for x in 0..width {
        let px = match header.color_type {
            0 => {
                let v = sample(row, x, depth).wrapping_mul(grey_scale(depth));
                [v, v, v]
            }
            2 => [
                sample(row, 3 * x, depth),
                sample(row, 3 * x + 1, depth),
                sample(row, 3 * x + 2, depth),
            ],
            3 => {
                let index = usize::from(sample(row, x, depth));
                *palette.get(index).ok_or_else(|| {
                    malformed_png(format!(
                        "palette index {index} past the {}-entry PLTE",
                        palette.len()
                    ))
                })?
            }
            4 => {
                let v = sample(row, 2 * x, depth);
                [v, v, v]
            }
            _ => [
                sample(row, 4 * x, depth),
                sample(row, 4 * x + 1, depth),
                sample(row, 4 * x + 2, depth),
            ],
        };
        out[3 * x..3 * x + 3].copy_from_slice(&px);
    }
    Ok(())
}

/// Decode a PNG (see the module docs).
///
/// # Errors
///
/// [`ImageInputError::Malformed`] for a structural problem (bad signature,
/// chunk, CRC, header, filter, palette index or data size),
/// [`ImageInputError::TooLarge`] / [`ImageInputError::Empty`] from the
/// header.
pub fn decode_png(bytes: &[u8]) -> Result<ImageRgb8, ImageInputError> {
    let body = bytes
        .strip_prefix(&PNG_SIGNATURE[..])
        .ok_or_else(|| malformed_png("missing the PNG signature"))?;
    let mut header: Option<PngHeader> = None;
    let mut palette: Vec<[u8; 3]> = Vec::new();
    let mut idat: Vec<u8> = Vec::new();
    let mut ended = false;
    let mut rest = body;
    while !rest.is_empty() {
        let (len_bytes, after) = rest
            .split_first_chunk::<4>()
            .ok_or_else(|| malformed_png("truncated chunk length"))?;
        let len = u32::from_be_bytes(*len_bytes) as usize;
        let (kind, after) = after
            .split_first_chunk::<4>()
            .ok_or_else(|| malformed_png("truncated chunk type"))?;
        let data = after.get(..len).ok_or_else(|| {
            malformed_png(format!("chunk {} runs past the end", chunk_name(kind)))
        })?;
        let crc_bytes = after
            .get(len..len + 4)
            .ok_or_else(|| malformed_png(format!("chunk {} lacks its CRC", chunk_name(kind))))?;
        rest = &after[len + 4..];
        // Upper-case first letter = critical chunk (PNG §5.4).
        let critical = kind[0].is_ascii_uppercase();
        if critical {
            let stored =
                u32::from_be_bytes([crc_bytes[0], crc_bytes[1], crc_bytes[2], crc_bytes[3]]);
            if crc32(&[kind, data]) != stored {
                return Err(malformed_png(format!(
                    "CRC mismatch in chunk {}",
                    chunk_name(kind)
                )));
            }
        }
        match kind {
            b"IHDR" => {
                if header.is_some() {
                    return Err(malformed_png("more than one IHDR"));
                }
                header = Some(PngHeader::parse(data)?);
            }
            b"PLTE" => {
                if data.is_empty() || data.len() % 3 != 0 || data.len() > 3 * 256 {
                    return Err(malformed_png(format!("PLTE of {} bytes", data.len())));
                }
                palette = data.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
            }
            b"IDAT" => {
                if header.is_none() {
                    return Err(malformed_png("IDAT before IHDR"));
                }
                if idat.len() + data.len() > MAX_ENCODED_IMAGE_BYTES {
                    return Err(ImageInputError::EncodedTooLarge {
                        bytes: idat.len() + data.len(),
                        limit: MAX_ENCODED_IMAGE_BYTES,
                    });
                }
                idat.extend_from_slice(data);
            }
            b"IEND" => {
                ended = true;
                break;
            }
            _ if critical => {
                return Err(malformed_png(format!(
                    "unknown critical chunk {}",
                    chunk_name(kind)
                )));
            }
            // Ancillary chunks (gamma, text, time, transparency, ...) do not
            // change RGB8 pixels as `stb_image` decodes them.
            _ => {}
        }
    }
    let header = header.ok_or_else(|| malformed_png("no IHDR"))?;
    if !ended {
        return Err(malformed_png("no IEND"));
    }
    if header.color_type == 3 && palette.is_empty() {
        return Err(malformed_png("palette image without PLTE"));
    }
    if idat.is_empty() {
        return Err(malformed_png("no IDAT"));
    }

    // The exact inflated size the header implies.
    let expected: usize = if header.interlaced {
        ADAM7
            .iter()
            .map(|&pass| {
                let (w, h) = pass_dims(header.width, header.height, pass);
                if w == 0 || h == 0 {
                    0
                } else {
                    h * (header.row_bytes(w) + 1)
                }
            })
            .sum()
    } else {
        header.height * (header.row_bytes(header.width) + 1)
    };
    let raw = inflate_zlib(&idat, expected)?;

    let (width, height) = (header.width, header.height);
    let mut rgb = vec![0u8; width * height * 3];
    if header.interlaced {
        let mut offset = 0usize;
        let mut line = vec![0u8; width * 3];
        for &pass in &ADAM7 {
            let (pw, ph) = pass_dims(width, height, pass);
            if pw == 0 || ph == 0 {
                continue;
            }
            let row_bytes = header.row_bytes(pw);
            let size = ph * (row_bytes + 1);
            let sub = raw
                .get(offset..offset + size)
                .ok_or_else(|| malformed_png("interlaced data ends early"))?;
            offset += size;
            let rows = unfilter(sub, ph, row_bytes, header.filter_bpp())?;
            let (x0, y0, dx, dy) = pass;
            for (j, row) in rows.chunks_exact(row_bytes).enumerate() {
                row_to_rgb(&header, &palette, row, pw, &mut line[..pw * 3])?;
                let y = y0 + j * dy;
                for i in 0..pw {
                    let x = x0 + i * dx;
                    let dst = (y * width + x) * 3;
                    rgb[dst..dst + 3].copy_from_slice(&line[i * 3..i * 3 + 3]);
                }
            }
        }
    } else {
        let row_bytes = header.row_bytes(width);
        let rows = unfilter(&raw, height, row_bytes, header.filter_bpp())?;
        for (y, row) in rows.chunks_exact(row_bytes).enumerate() {
            row_to_rgb(
                &header,
                &palette,
                row,
                width,
                &mut rgb[y * width * 3..(y + 1) * width * 3],
            )?;
        }
    }
    ImageRgb8::new(width, height, rgb).map_err(|e| malformed_png(e.to_string()))
}

/// A chunk type for messages (printable ASCII, else hex).
fn chunk_name(kind: &[u8; 4]) -> String {
    if kind.iter().all(u8::is_ascii_alphabetic) {
        String::from_utf8_lossy(kind).into_owned()
    } else {
        format!(
            "{:02x}{:02x}{:02x}{:02x}",
            kind[0], kind[1], kind[2], kind[3]
        )
    }
}

/// Inflate a PNG's concatenated `IDAT` payload into exactly `expected`
/// bytes.
///
/// The zlib header is checked here; the DEFLATE stream is inflated straight
/// into a buffer of the size the image header implies, so a stream that
/// would inflate further is refused instead of allocated. Like
/// `stb_image` and the `png` crate's defaults, the Adler-32 trailer (and
/// any padding after the stream) is not required.
fn inflate_zlib(idat: &[u8], expected: usize) -> Result<Vec<u8>, ImageInputError> {
    let [cmf, flg, ..] = *idat else {
        return Err(malformed_png("zlib stream shorter than its header"));
    };
    if cmf & 0x0F != 8 || cmf >> 4 > 7 {
        return Err(malformed_png(format!(
            "zlib header {cmf:02x}{flg:02x}: not a DEFLATE stream"
        )));
    }
    if (u16::from(cmf) * 256 + u16::from(flg)) % 31 != 0 {
        return Err(malformed_png("zlib header check bits are wrong"));
    }
    if flg & 0x20 != 0 {
        return Err(malformed_png("zlib preset dictionary (not allowed in PNG)"));
    }
    let mut raw = vec![0u8; expected];
    let written = oxiarc_deflate::inflate_into(&idat[2..], &mut raw).map_err(|e| {
        malformed_png(format!(
            "image data does not inflate to the {expected} bytes its header declares: {e}"
        ))
    })?;
    if written != expected {
        return Err(malformed_png(format!(
            "image data inflates to {written} bytes; its header declares {expected}"
        )));
    }
    Ok(raw)
}

// ─────────────────────────────────────────────────────────────────────────
//  JPEG
// ─────────────────────────────────────────────────────────────────────────

/// `stb_image`'s `stbi__blinn_8x8`: `x * y / 255`, rounded.
fn blinn_8x8(x: u8, y: u8) -> u8 {
    let t = u32::from(x) * u32::from(y) + 128;
    ((t + (t >> 8)) >> 8) as u8
}

fn jpeg_error(error: jpeg_decoder::Error) -> ImageInputError {
    match error {
        jpeg_decoder::Error::Unsupported(feature) => ImageInputError::Unsupported {
            format: "JPEG",
            reason: format!("{feature:?}"),
        },
        other => ImageInputError::Malformed {
            format: "JPEG",
            reason: other.to_string(),
        },
    }
}

/// Decode a baseline or progressive JPEG (see the module docs).
///
/// # Errors
///
/// [`ImageInputError::Malformed`] / [`ImageInputError::Unsupported`] from
/// the decoder, [`ImageInputError::TooLarge`] / [`ImageInputError::Empty`]
/// from the frame header (checked before any pixel buffer exists).
pub fn decode_jpeg(bytes: &[u8]) -> Result<ImageRgb8, ImageInputError> {
    // A third-party decoder on caller bytes: a panic inside it must become
    // a typed error, never take the process down.
    let result = std::panic::catch_unwind(|| decode_jpeg_inner(bytes));
    result.unwrap_or_else(|_| {
        Err(ImageInputError::Malformed {
            format: "JPEG",
            reason: "the decoder rejected the data".to_string(),
        })
    })
}

fn decode_jpeg_inner(bytes: &[u8]) -> Result<ImageRgb8, ImageInputError> {
    let mut decoder = jpeg_decoder::Decoder::new(bytes);
    decoder.read_info().map_err(jpeg_error)?;
    let info = decoder.info().ok_or_else(|| ImageInputError::Malformed {
        format: "JPEG",
        reason: "no frame header".to_string(),
    })?;
    let (width, height) = (usize::from(info.width), usize::from(info.height));
    check_dimensions(width, height)?;
    let pixel_bytes = info.pixel_format.pixel_bytes();
    decoder.set_max_decoding_buffer_size(width * height * pixel_bytes);
    let pixels = decoder.decode().map_err(jpeg_error)?;
    let n = width * height;
    if pixels.len() != n * pixel_bytes {
        return Err(ImageInputError::Malformed {
            format: "JPEG",
            reason: format!(
                "decoded {} bytes for a {width} x {height} frame",
                pixels.len()
            ),
        });
    }
    let rgb: Vec<u8> = match info.pixel_format {
        jpeg_decoder::PixelFormat::RGB24 => pixels,
        jpeg_decoder::PixelFormat::L8 => pixels.iter().flat_map(|&v| [v, v, v]).collect(),
        jpeg_decoder::PixelFormat::CMYK32 => pixels
            .chunks_exact(4)
            .flat_map(|px| {
                // The decoder returns true CMYK (it undoes Adobe's inverted
                // storage; for YCCK, R'G'B' plus 255 - K). `stb_image`
                // computes blinn(stored, stored K) on the inverted values,
                // i.e. blinn(255 - c, 255 - k) here — for both kinds.
                let k = 255 - px[3];
                [
                    blinn_8x8(255 - px[0], k),
                    blinn_8x8(255 - px[1], k),
                    blinn_8x8(255 - px[2], k),
                ]
            })
            .collect(),
        jpeg_decoder::PixelFormat::L16 => {
            return Err(ImageInputError::Unsupported {
                format: "JPEG",
                reason: "12/16-bit sample precision (the reference decoder refuses it too)"
                    .to_string(),
            })
        }
    };
    ImageRgb8::new(width, height, rgb).map_err(|e| ImageInputError::Malformed {
        format: "JPEG",
        reason: e.to_string(),
    })
}

// ─────────────────────────────────────────────────────────────────────────
//  data: URIs and base64
// ─────────────────────────────────────────────────────────────────────────

fn data_uri_error(reason: impl Into<String>) -> ImageInputError {
    ImageInputError::DataUri {
        reason: reason.into(),
    }
}

/// Decode standard (`+/`) or URL-safe (`-_`) base64, with or without `=`
/// padding, ignoring ASCII whitespace; anything else is an error (never a
/// silent truncation).
///
/// # Errors
///
/// [`ImageInputError::DataUri`] naming the offending character or length,
/// [`ImageInputError::EncodedTooLarge`] above [`MAX_ENCODED_IMAGE_BYTES`].
pub fn decode_base64(text: &str) -> Result<Vec<u8>, ImageInputError> {
    let mut out = Vec::with_capacity(text.len() / 4 * 3 + 3);
    let mut acc = 0u32;
    let mut bits = 0u32;
    let mut padding = 0usize;
    let mut symbols = 0usize;
    for (i, c) in text.bytes().enumerate() {
        if c.is_ascii_whitespace() {
            continue;
        }
        if c == b'=' {
            padding += 1;
            continue;
        }
        if padding > 0 {
            return Err(data_uri_error(format!(
                "base64 data continues after '=' padding (offset {i})"
            )));
        }
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            other => {
                return Err(data_uri_error(format!(
                    "invalid base64 character {:?} at offset {i}",
                    char::from(other)
                )))
            }
        };
        acc = (acc << 6) | u32::from(v);
        bits += 6;
        symbols += 1;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
        if out.len() > MAX_ENCODED_IMAGE_BYTES {
            return Err(ImageInputError::EncodedTooLarge {
                bytes: out.len(),
                limit: MAX_ENCODED_IMAGE_BYTES,
            });
        }
    }
    if symbols % 4 == 1 {
        return Err(data_uri_error(format!(
            "base64 data has {symbols} symbols, which no byte count encodes"
        )));
    }
    if padding > 2 || (padding > 0 && !(symbols + padding).is_multiple_of(4)) {
        return Err(data_uri_error("malformed base64 '=' padding"));
    }
    if out.is_empty() {
        return Err(data_uri_error("empty base64 payload"));
    }
    Ok(out)
}

/// A parsed `data:` URI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataUri {
    /// The declared media type (`image/png`, ...), empty when omitted.
    pub media_type: String,
    /// The decoded payload.
    pub bytes: Vec<u8>,
}

/// Parse a base64 `data:` URI (`data:[<media type>][;params];base64,<data>`).
///
/// A declared media type other than `image/*` is refused (the payload is a
/// document, not a picture); the image format itself is still sniffed from
/// the bytes, so a mislabelled PNG decodes.
///
/// # Errors
///
/// [`ImageInputError::DataUri`] for a missing `data:` scheme or `,`, a
/// non-base64 encoding, a non-image media type or bad base64;
/// [`ImageInputError::EncodedTooLarge`] for an oversized payload.
pub fn parse_data_uri(uri: &str) -> Result<DataUri, ImageInputError> {
    let rest = strip_prefix_ignore_case(uri, "data:")
        .ok_or_else(|| data_uri_error("the reference does not start with 'data:'"))?;
    let (meta, payload) = rest
        .split_once(',')
        .ok_or_else(|| data_uri_error("no ',' between the header and the payload"))?;
    let mut params = meta.split(';');
    let media_type = params
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    let base64 = params.any(|p| p.trim().eq_ignore_ascii_case("base64"));
    if !base64 {
        return Err(data_uri_error(
            "only base64 data URIs are accepted (data:image/...;base64,...)",
        ));
    }
    if !media_type.is_empty() && !media_type.starts_with("image/") {
        return Err(data_uri_error(format!(
            "media type {media_type:?} is not an image"
        )));
    }
    // Refuse an oversized payload before decoding any of it.
    if payload.len() / 4 * 3 > MAX_ENCODED_IMAGE_BYTES + 3 {
        return Err(ImageInputError::EncodedTooLarge {
            bytes: payload.len() / 4 * 3,
            limit: MAX_ENCODED_IMAGE_BYTES,
        });
    }
    Ok(DataUri {
        media_type,
        bytes: decode_base64(payload)?,
    })
}

fn strip_prefix_ignore_case<'s>(s: &'s str, prefix: &str) -> Option<&'s str> {
    let head = s.get(..prefix.len())?;
    head.eq_ignore_ascii_case(prefix)
        .then(|| &s[prefix.len()..])
}

// ─────────────────────────────────────────────────────────────────────────
//  Image references
// ─────────────────────────────────────────────────────────────────────────

/// Which image references [`load_image_source`] resolves.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ImageSourcePolicy {
    /// Plain local paths (and `file://` URLs) are read as given — for a
    /// command-line user naming their own files.
    pub allow_any_local_path: bool,
    /// A directory `file://` references resolve inside (`OXI_MEDIA_PATH`
    /// for `oxibonsai serve`, like the reference server's `--media-path`):
    /// no absolute path, no `..`, and the resolved file must stay inside
    /// it. `None` with `allow_any_local_path == false` refuses local files
    /// — a network server's default.
    pub media_root: Option<PathBuf>,
    /// The operator opted in to remote `http(s)` references. They are
    /// still refused (no fetcher with an address policy exists yet), with a
    /// reason that says so.
    pub allow_remote_fetch: bool,
}

impl ImageSourcePolicy {
    /// The policy for a command-line user's own references: local files
    /// allowed, no remote fetch.
    #[must_use]
    pub fn local_user() -> Self {
        Self {
            allow_any_local_path: true,
            media_root: None,
            allow_remote_fetch: false,
        }
    }

    /// The policy for a network server: `data:` URIs only, unless the
    /// operator named a media directory (`file://` references inside it)
    /// or opted in to remote references.
    #[must_use]
    pub fn server(media_root: Option<PathBuf>, allow_remote_fetch: bool) -> Self {
        Self {
            allow_any_local_path: false,
            media_root,
            allow_remote_fetch,
        }
    }
}

/// How an image reference was classified.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageSource {
    /// A `data:` URI.
    DataUri,
    /// A local file reference (`file://` or a plain path).
    LocalFile(String),
    /// An `http(s)` URL.
    Remote,
}

/// Classify an image reference without resolving it.
///
/// # Errors
///
/// [`ImageInputError::UnsupportedSource`] for an empty reference or a
/// scheme other than `data`, `file`, `http` and `https`.
pub fn classify_image_source(reference: &str) -> Result<ImageSource, ImageInputError> {
    let r = reference.trim();
    if r.is_empty() {
        return Err(ImageInputError::UnsupportedSource {
            reason: "empty image reference".to_string(),
        });
    }
    if strip_prefix_ignore_case(r, "data:").is_some() {
        return Ok(ImageSource::DataUri);
    }
    if strip_prefix_ignore_case(r, "http://").is_some()
        || strip_prefix_ignore_case(r, "https://").is_some()
    {
        return Ok(ImageSource::Remote);
    }
    if let Some(path) = strip_prefix_ignore_case(r, "file://") {
        return Ok(ImageSource::LocalFile(path.to_string()));
    }
    // `scheme:` with a scheme of two or more letters is a URL of a kind this
    // build does not resolve (a single letter is a Windows drive).
    if let Some((scheme, _)) = r.split_once(':') {
        let looks_like_scheme = scheme.len() > 1
            && scheme
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
            && scheme
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_alphabetic());
        if looks_like_scheme {
            return Err(ImageInputError::UnsupportedSource {
                reason: format!("the '{scheme}:' scheme is not resolved (use data:, or a file)"),
            });
        }
    }
    Ok(ImageSource::LocalFile(r.to_string()))
}

/// Read a local file of at most [`MAX_ENCODED_IMAGE_BYTES`].
fn read_limited(path: &Path, shown: &str) -> Result<Vec<u8>, ImageInputError> {
    let unreadable = |reason: String| ImageInputError::FileUnreadable {
        path: shown.to_string(),
        reason,
    };
    let meta = std::fs::metadata(path).map_err(|e| unreadable(e.to_string()))?;
    if !meta.is_file() {
        return Err(unreadable("not a regular file".to_string()));
    }
    let len = usize::try_from(meta.len()).unwrap_or(usize::MAX);
    if len > MAX_ENCODED_IMAGE_BYTES {
        return Err(ImageInputError::EncodedTooLarge {
            bytes: len,
            limit: MAX_ENCODED_IMAGE_BYTES,
        });
    }
    std::fs::read(path).map_err(|e| unreadable(e.to_string()))
}

/// Resolve a `file://` reference inside `root`: relative, no `..`, and
/// still inside `root` once symlinks are resolved.
fn resolve_in_media_root(root: &Path, relative: &str) -> Result<PathBuf, ImageInputError> {
    let refused = |reason: String| ImageInputError::LocalFileRefused { reason };
    let rel = Path::new(relative);
    if relative.is_empty()
        || rel.is_absolute()
        || rel
            .components()
            .any(|c| !matches!(c, Component::Normal(_) | Component::CurDir))
    {
        return Err(refused(format!(
            "{relative:?} must be a relative path inside the media directory, without '..'"
        )));
    }
    let root = root
        .canonicalize()
        .map_err(|e| refused(format!("the media directory cannot be resolved: {e}")))?;
    let candidate = root.join(rel);
    let resolved = candidate
        .canonicalize()
        .map_err(|e| ImageInputError::FileUnreadable {
            path: relative.to_string(),
            reason: e.to_string(),
        })?;
    if !resolved.starts_with(&root) {
        return Err(refused(format!(
            "{relative:?} resolves outside the media directory"
        )));
    }
    Ok(resolved)
}

/// Resolve an image reference to its encoded bytes under `policy` (see the
/// module docs), without decoding it.
///
/// # Errors
///
/// [`ImageInputError`]: a malformed data URI, a refused or unreadable local
/// file, a refused remote URL, an unsupported scheme or an oversized input.
pub fn load_image_bytes(
    reference: &str,
    policy: &ImageSourcePolicy,
) -> Result<Vec<u8>, ImageInputError> {
    let reference = reference.trim();
    match classify_image_source(reference)? {
        ImageSource::DataUri => parse_data_uri(reference).map(|uri| uri.bytes),
        ImageSource::Remote => {
            let shown: String = reference.chars().take(96).collect();
            Err(ImageInputError::RemoteFetchRefused {
                url: shown,
                reason: if policy.allow_remote_fetch {
                    "remote fetching was enabled, but this build has no image fetcher with an \
                     address policy (loopback/private ranges, redirects, size and time limits)"
                        .to_string()
                } else {
                    "fetching them would let a request make this process open arbitrary \
                     network connections (server-side request forgery); remote image URLs are \
                     disabled unless the operator opts in (OXI_ALLOW_IMAGE_URL_FETCH=1)"
                        .to_string()
                },
            })
        }
        ImageSource::LocalFile(path) => {
            if policy.allow_any_local_path {
                return read_limited(Path::new(&path), &path);
            }
            match &policy.media_root {
                Some(root) => {
                    let resolved = resolve_in_media_root(root, &path)?;
                    read_limited(&resolved, &path)
                }
                None => Err(ImageInputError::LocalFileRefused {
                    reason: format!(
                        "{path:?}: local files are not served unless the operator names a media \
                         directory (OXI_MEDIA_PATH); send the image as a base64 data URI"
                    ),
                }),
            }
        }
    }
}

/// [`load_image_bytes`] followed by [`decode_image`].
///
/// # Errors
///
/// As the two calls it makes.
pub fn load_image_source(
    reference: &str,
    policy: &ImageSourcePolicy,
) -> Result<ImageRgb8, ImageInputError> {
    decode_image(&load_image_bytes(reference, policy)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("test vector hex"))
            .collect()
    }

    fn decode_ok(png_hex: &str, rgb_hex: &str, dims: (usize, usize)) {
        let img = decode_image(&hex(png_hex)).expect("decodes");
        assert_eq!((img.width, img.height), dims);
        assert_eq!(img.data, hex(rgb_hex));
    }

    /// Every colour type and legal bit depth, per-row filters 0..=4,
    /// split `IDAT`, ancillary chunks — from an independent encoder, with
    /// the expected RGB computed the way `stb_image` converts (alpha
    /// dropped, grey replicated, sub-byte grey scaled, 16-bit high byte).
    #[test]
    fn png_every_colour_type_and_bit_depth_decodes_like_stb_image() {
        decode_ok(PNG_RGB8, PNG_RGB8_RGB, PNG_RGB8_DIMS);
        decode_ok(PNG_RGBA8, PNG_RGBA8_RGB, PNG_RGBA8_DIMS);
        decode_ok(PNG_RGB16, PNG_RGB16_RGB, PNG_RGB16_DIMS);
        decode_ok(PNG_RGBA16, PNG_RGBA16_RGB, PNG_RGBA16_DIMS);
        decode_ok(PNG_GREY1, PNG_GREY1_RGB, PNG_GREY1_DIMS);
        decode_ok(PNG_GREY2, PNG_GREY2_RGB, PNG_GREY2_DIMS);
        decode_ok(PNG_GREY4, PNG_GREY4_RGB, PNG_GREY4_DIMS);
        decode_ok(PNG_GREY8, PNG_GREY8_RGB, PNG_GREY8_DIMS);
        decode_ok(PNG_GREY16, PNG_GREY16_RGB, PNG_GREY16_DIMS);
        decode_ok(PNG_GREYA8, PNG_GREYA8_RGB, PNG_GREYA8_DIMS);
        decode_ok(PNG_GREYA16, PNG_GREYA16_RGB, PNG_GREYA16_DIMS);
        decode_ok(PNG_PAL1, PNG_PAL1_RGB, PNG_PAL1_DIMS);
        decode_ok(PNG_PAL2, PNG_PAL2_RGB, PNG_PAL2_DIMS);
        decode_ok(PNG_PAL4, PNG_PAL4_RGB, PNG_PAL4_DIMS);
        decode_ok(PNG_PAL8, PNG_PAL8_RGB, PNG_PAL8_DIMS);
    }

    #[test]
    fn png_adam7_interlacing_decodes() {
        decode_ok(PNG_ADAM7_RGB8, PNG_ADAM7_RGB8_RGB, PNG_ADAM7_RGB8_DIMS);
        decode_ok(PNG_ADAM7_GREY1, PNG_ADAM7_GREY1_RGB, PNG_ADAM7_GREY1_DIMS);
        decode_ok(PNG_ADAM7_PAL4, PNG_ADAM7_PAL4_RGB, PNG_ADAM7_PAL4_DIMS);
        decode_ok(PNG_ADAM7_TINY, PNG_ADAM7_TINY_RGB, PNG_ADAM7_TINY_DIMS);
    }

    /// Flip one byte at `at` and re-seal nothing: the decoder must refuse
    /// the result with a typed error, never panic.
    fn corrupted(bytes: &[u8], at: usize) -> Vec<u8> {
        let mut out = bytes.to_vec();
        out[at] ^= 0x5A;
        out
    }

    #[test]
    fn png_corruption_is_a_typed_error_never_a_panic() {
        let good = hex(PNG_RGB8);
        // Every single-byte corruption and every truncation.
        for at in 0..good.len() {
            let _ = decode_image(&corrupted(&good, at));
        }
        for len in 0..good.len() {
            let result = decode_image(&good[..len]);
            assert!(result.is_err(), "a PNG truncated to {len} bytes decoded");
        }
        // A header CRC flip is named.
        let err = decode_image(&corrupted(&good, 20)).expect_err("IHDR corrupted");
        assert_eq!(err.code(), "image_decode_failed");
        assert!(err.to_string().contains("CRC"), "{err}");
    }

    #[test]
    fn png_oversized_header_is_refused_before_allocation() {
        // 40000 x 40000 RGB: refused from IHDR alone.
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&40_000u32.to_be_bytes());
        ihdr.extend_from_slice(&40_000u32.to_be_bytes());
        ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
        let mut png = PNG_SIGNATURE.to_vec();
        png.extend_from_slice(&13u32.to_be_bytes());
        png.extend_from_slice(b"IHDR");
        png.extend_from_slice(&ihdr);
        png.extend_from_slice(&crc32(&[b"IHDR", &ihdr]).to_be_bytes());
        let err = decode_image(&png).expect_err("too large");
        assert_eq!(err.code(), "image_too_large");
        // A zero side.
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&0u32.to_be_bytes());
        ihdr.extend_from_slice(&4u32.to_be_bytes());
        ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
        let mut png = PNG_SIGNATURE.to_vec();
        png.extend_from_slice(&13u32.to_be_bytes());
        png.extend_from_slice(b"IHDR");
        png.extend_from_slice(&ihdr);
        png.extend_from_slice(&crc32(&[b"IHDR", &ihdr]).to_be_bytes());
        assert_eq!(decode_image(&png).expect_err("empty").code(), "image_empty");
    }

    #[test]
    fn crc32_matches_the_png_reference_value() {
        // The CRC of the IEND chunk type, a constant every PNG ends with.
        assert_eq!(crc32(&[b"IEND"]), 0xAE42_6082);
        assert_eq!(crc32(&[b"123456789"]), 0xCBF4_3926);
    }

    fn max_abs_diff(a: &[u8], b: &[u8]) -> u8 {
        a.iter()
            .zip(b)
            .map(|(x, y)| x.abs_diff(*y))
            .max()
            .unwrap_or(0)
    }

    /// Baseline (4:2:0 and 4:4:4), progressive, greyscale and Adobe CMYK
    /// JPEGs against libjpeg's (Pillow's) decode of the same bytes: IDCT
    /// and upsampling differ by at most a couple of units per channel.
    #[test]
    fn jpeg_baseline_progressive_grey_and_cmyk_decode() {
        for (name, jpeg, want, tolerance) in [
            ("baseline", JPEG_BASELINE, JPEG_BASELINE_PIL_RGB, 3u8),
            ("progressive", JPEG_PROGRESSIVE, JPEG_PROGRESSIVE_PIL_RGB, 3),
            ("4:4:4", JPEG_444, JPEG_444_PIL_RGB, 3),
            ("grey", JPEG_GREY, JPEG_GREY_PIL_RGB, 2),
            ("cmyk", JPEG_CMYK, JPEG_CMYK_PIL_RGB, 4),
        ] {
            let img = decode_image(&hex(jpeg)).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!((img.width, img.height), (16, 16), "{name}");
            let diff = max_abs_diff(&img.data, &hex(want));
            assert!(diff <= tolerance, "{name}: max |diff| {diff} vs libjpeg");
        }
        // The progressive and baseline encodings of the same picture agree
        // with each other just as closely.
        let a = decode_image(&hex(JPEG_BASELINE)).expect("baseline");
        let b = decode_image(&hex(JPEG_PROGRESSIVE)).expect("progressive");
        assert!(max_abs_diff(&a.data, &b.data) <= 4);
    }

    #[test]
    fn jpeg_corruption_is_a_typed_error_never_a_panic() {
        let good = hex(JPEG_PROGRESSIVE);
        for len in 3..good.len() {
            let _ = decode_image(&good[..len]);
        }
        for at in (2..good.len()).step_by(7) {
            let _ = decode_image(&corrupted(&good, at));
        }
        let err = decode_image(&good[..40]).expect_err("truncated");
        assert!(
            matches!(
                err.code(),
                "image_decode_failed" | "image_format_unsupported"
            ),
            "{err}"
        );
    }

    #[test]
    fn unknown_formats_are_named() {
        let gif = decode_image(b"GIF89a\x01\x00\x01\x00").expect_err("gif");
        assert_eq!(gif.code(), "image_format_unsupported");
        assert!(gif.to_string().contains("GIF"), "{gif}");
        let webp = decode_image(b"RIFF\0\0\0\0WEBPVP8 ").expect_err("webp");
        assert!(webp.to_string().contains("WebP"), "{webp}");
        let empty = decode_image(&[]).expect_err("empty");
        assert!(empty.to_string().contains("empty"), "{empty}");
    }

    #[test]
    fn base64_accepts_padding_whitespace_and_url_safe_symbols() {
        assert_eq!(decode_base64("aGVsbG8=").expect("padded"), b"hello");
        assert_eq!(decode_base64("aGVsbG8").expect("unpadded"), b"hello");
        assert_eq!(decode_base64("aGVs\nbG8=").expect("wrapped"), b"hello");
        assert_eq!(
            decode_base64("-_-_").expect("url-safe"),
            vec![0xFB, 0xFF, 0xBF]
        );
        assert_eq!(
            decode_base64("+/+/").expect("standard"),
            vec![0xFB, 0xFF, 0xBF]
        );
        for bad in ["aGVsbG8*", "aGVsb=G8", "a", "aGVsbG8===", "", "===="] {
            let err = decode_base64(bad).expect_err(bad);
            assert_eq!(err.code(), "image_data_uri_invalid", "{bad}: {err}");
        }
    }

    #[test]
    fn data_uris_are_parsed_strictly() {
        let png = hex(PNG_RGB8);
        let b64 = encode_base64_for_test(&png);
        let uri = format!("data:image/png;base64,{b64}");
        let parsed = parse_data_uri(&uri).expect("data uri");
        assert_eq!(parsed.media_type, "image/png");
        assert_eq!(parsed.bytes, png);
        // Scheme case, extra parameters and an omitted media type are fine;
        // a JPEG label on PNG bytes still decodes (the format is sniffed).
        let loose = format!("DATA:;charset=x;base64,{b64}");
        assert_eq!(parse_data_uri(&loose).expect("loose").bytes, png);
        let mislabelled = format!("data:image/jpeg;base64,{b64}");
        let img = load_image_source(&mislabelled, &ImageSourcePolicy::default()).expect("sniffed");
        assert_eq!(img.data, hex(PNG_RGB8_RGB));
        for (bad, why) in [
            ("data:image/png,rawbytes", "not base64"),
            ("data:text/plain;base64,aGVsbG8=", "not an image"),
            ("data:image/png;base64", "no comma"),
            ("image/png;base64,aGVsbG8=", "no scheme"),
        ] {
            let err = parse_data_uri(bad).expect_err(why);
            assert_eq!(err.code(), "image_data_uri_invalid", "{why}: {err}");
        }
    }

    /// A minimal standard base64 encoder for the tests (the crate itself
    /// only decodes).
    fn encode_base64_for_test(bytes: &[u8]) -> String {
        const ALPHABET: &[u8; 64] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for chunk in bytes.chunks(3) {
            let n = chunk.len();
            let v = (u32::from(chunk[0]) << 16)
                | (u32::from(*chunk.get(1).unwrap_or(&0)) << 8)
                | u32::from(*chunk.get(2).unwrap_or(&0));
            for i in 0..4 {
                if i <= n {
                    out.push(char::from(ALPHABET[((v >> (18 - 6 * i)) & 63) as usize]));
                } else {
                    out.push('=');
                }
            }
        }
        out
    }

    #[test]
    fn sources_follow_the_policy() {
        let dir = std::env::temp_dir().join(format!(
            "oxibonsai-image-source-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(dir.join("sub")).expect("temp dir");
        let file = dir.join("sub").join("pic.png");
        std::fs::write(&file, hex(PNG_RGB8)).expect("write fixture");
        let path = file.to_string_lossy().into_owned();

        // The CLI: any local path, as given or as a file:// URL.
        let cli = ImageSourcePolicy::local_user();
        assert_eq!(load_image_source(&path, &cli).expect("path").width, 5);
        let file_url = format!("file://{path}");
        assert_eq!(
            load_image_source(&file_url, &cli).expect("file url").width,
            5
        );

        // A server without a media directory: data URIs only.
        let server = ImageSourcePolicy::server(None, false);
        let err = load_image_source(&path, &server).expect_err("refused");
        assert_eq!(err.code(), "image_file_refused");

        // A server with one: relative file:// references inside it.
        let rooted = ImageSourcePolicy::server(Some(dir.clone()), false);
        let ok = load_image_source("file://sub/pic.png", &rooted).expect("inside the root");
        assert_eq!(ok.height, 4);
        for escape in ["file://../pic.png", "file://sub/../../x.png", &file_url] {
            let err = load_image_source(escape, &rooted).expect_err(escape);
            assert_eq!(err.code(), "image_file_refused", "{escape}: {err}");
        }
        let missing = load_image_source("file://sub/none.png", &rooted).expect_err("missing");
        assert_eq!(missing.code(), "image_file_unreadable");

        // Remote URLs: refused with or without the opt-in, naming why.
        for policy in [&cli, &server, &ImageSourcePolicy::server(None, true)] {
            let err = load_image_source("https://example.com/cat.png", policy).expect_err("remote");
            assert_eq!(err.code(), "image_url_fetch_disabled");
        }
        let err = load_image_source("ftp://example.com/cat.png", &cli).expect_err("ftp");
        assert_eq!(err.code(), "image_url_scheme_unsupported");
        let err = load_image_source("   ", &cli).expect_err("blank");
        assert_eq!(err.code(), "image_url_scheme_unsupported");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn classification_distinguishes_drives_from_schemes() {
        assert_eq!(
            classify_image_source("C:\\pics\\a.png").expect("drive"),
            ImageSource::LocalFile("C:\\pics\\a.png".to_string())
        );
        assert_eq!(
            classify_image_source("data:image/png;base64,AA").expect("data"),
            ImageSource::DataUri
        );
        assert_eq!(
            classify_image_source("HTTPS://x").expect("remote"),
            ImageSource::Remote
        );
    }

    // ── Test vectors (PNG from an independent encoder, JPEG + reference decode from Pillow) ──
    const PNG_RGB8: &str = "89504e470d0a1a0a0000000d4948445200000005000000040802000000c95162170000000e74455874436f6d6d656e74\
         00766563746f72236665970000000467414d410000b18f0bfc6105000000254944415478da014000bfff00c394ed4bdf\
         79bbf622049fd8ebe3e501431c5fb5c2e167a4619dc863465b5dcf7a000000264944415499e5027b7a08e50320b73d69\
         d9397868fb1103991bd474b01222a5164acdc28f002131a01e813c7f08250000000049454e44ae426082";
    const PNG_RGB8_RGB: &str = "c394ed4bdf79bbf622049fd8ebe3e5431c5ff8de405f82a1fc4a0442e3e9be9667dde16016bf0ad5837caadefaf86607\
         5e53455c2e3de2251e5581ad";
    const PNG_RGB8_DIMS: (usize, usize) = (5, 4);
    const PNG_RGBA8: &str = "89504e470d0a1a0a0000000d49484452000000030000000308060000005628b5bf0000000e74455874436f6d6d656e74\
         00766563746f72236665970000000467414d410000b18f0bfc6105000000194944415478da012700d8ff00746452439a\
         d19b4d0931da49019af9351e91272fdf0000001949444154b225535af3d7ff0c023d880e88f8dc9c5feb37978a473d11\
         cdec9da0b40000000049454e44ae426082";
    const PNG_RGBA8_RGB: &str = "7464529ad19b0931da9af9354c1e883ff587d7814344fa242a2c1e";
    const PNG_RGBA8_DIMS: (usize, usize) = (3, 3);
    const PNG_RGB16: &str = "89504e470d0a1a0a0000000d49484452000000040000000310020000006b06e5d20000000e74455874436f6d6d656e74\
         00766563746f72236665970000000467414d410000b18f0bfc61050000002b4944415478da014b00b4ff007d09a7bc24\
         6ad92a2190b1e69009f4d34a48d80e938edbf3017f7e69429946cfe83ddc53198ea50000002b49444154b908bb765a8d\
         8132af2154a99cf002dbf62e96e3660cc113e3519347c523ad7bd3c510362bf1206c1624bfabe6f8940000000049454e\
         44ae426082";
    const PNG_RGB16_RGB: &str =
        "7da724d921b190f44ad893db7f69994ea6520900d3b8546f5a977c5ab9a350234e7d8a60";
    const PNG_RGB16_DIMS: (usize, usize) = (4, 3);
    const PNG_RGBA16: &str = "89504e470d0a1a0a0000000d4948445200000002000000031006000000e97a02c20000000e74455874436f6d6d656e74\
         00766563746f72236665970000000467414d410000b18f0bfc61050000001f4944415478da013300ccff00132b220f71\
         3018cde900250a4c8ef4a90145d135fe39a712f671990000001f49444154f75328d41a48fbb9ff810256bc413367c9be\
         3c4d31ab0dda611c46fbf9151005feba750000000049454e44ae426082";
    const PNG_RGBA16_RGB: &str = "132271e9254c4535396d4f349b76a0bafa0e";
    const PNG_RGBA16_DIMS: (usize, usize) = (2, 3);
    const PNG_GREY1: &str = "89504e470d0a1a0a0000000d4948445200000009000000020100000000a22dcb7e0000000e74455874436f6d6d656e74\
         00766563746f72236665970000000467414d410000b18f0bfc6105000000074944415478da63f0686064146635650000\
         000749444154af040003f8014a45542efc0000000049454e44ae426082";
    const PNG_GREY1_RGB: &str = "000000ffffff000000000000ffffff000000000000000000ffffff000000000000000000000000000000ffffffffffff\
         ffffffffffff";
    const PNG_GREY1_DIMS: (usize, usize) = (9, 2);
    const PNG_GREY2: &str = "89504e470d0a1a0a0000000d494844520000000500000003020000000034ed82850000000e74455874436f6d6d656e74\
         00766563746f72236665970000000467414d410000b18f0bfc6105000000084944415478da63887360bc96f1f91e9900\
         00000949444154c564ec00000b4102554caf32790000000049454e44ae426082";
    const PNG_GREY2_RGB: &str = "555555555555ffffffaaaaaa555555ffffff555555555555aaaaaa555555000000000000aaaaaa555555aaaaaa";
    const PNG_GREY2_DIMS: (usize, usize) = (5, 3);
    const PNG_GREY4: &str = "89504e470d0a1a0a0000000d49484452000000030000000204000000007defd4c70000000e74455874436f6d6d656e74\
         00766563746f72236665970000000467414d410000b18f0bfc6105000000074944415478da6378b181f179d300680000\
         000749444154c202000991030a153dd31d0000000049454e44ae426082";
    const PNG_GREY4_RGB: &str = "eeeeee888888bbbbbbdddddd000000777777";
    const PNG_GREY4_DIMS: (usize, usize) = (3, 2);
    const PNG_GREY8: &str = "89504e470d0a1a0a0000000d49484452000000040000000208000000005ac322bf0000000e74455874436f6d6d656e74\
         00766563746f72236665970000000467414d410000b18f0bfc6105000000094944415478da63b8c37ec690f13add9aec\
         0000000949444154d47dee7b0015760474e94bc6010000000049454e44ae426082";
    const PNG_GREY8_RGB: &str = "dcdcdc070707cccccc313131cacacaa9a9a9b4b4b4929292";
    const PNG_GREY8_DIMS: (usize, usize) = (4, 2);
    const PNG_GREY16: &str = "89504e470d0a1a0a0000000d4948445200000003000000021000000000e88fe5850000000e74455874436f6d6d656e74\
         00766563746f72236665970000000467414d410000b18f0bfc61050000000b4944415478da63a85d726a575f2123b415\
         a27a0000000b4944415457564bcb933b002f2406e29d7b28180000000049454e44ae426082";
    const PNG_GREY16_RGB: &str = "7d7d7dcacaca8e8e8e0a0a0a8e8e8e727272";
    const PNG_GREY16_DIMS: (usize, usize) = (3, 2);
    const PNG_GREYA8: &str = "89504e470d0a1a0a0000000d4948445200000002000000020804000000d8bfc5af0000000e74455874436f6d6d656e74\
         00766563746f72236665970000000467414d410000b18f0bfc6105000000094944415478da6310b8b06c37a367ce83c4\
         0000000949444154dae55bda0014fb04417a033b0e0000000049454e44ae426082";
    const PNG_GREYA8_RGB: &str = "101010a6a6a6262626000000";
    const PNG_GREYA8_DIMS: (usize, usize) = (2, 2);
    const PNG_GREYA16: &str = "89504e470d0a1a0a0000000d4948445200000003000000011004000000e179007c0000000e74455874436f6d6d656e74\
         00766563746f72236665970000000467414d410000b18f0bfc61050000000a4944415478da6348fe37a774eab723bf8d\
         790000000b49444154d742b62eac8d002d590602acc4c7fe0000000049454e44ae426082";
    const PNG_GREYA16_RGB: &str = "6363639595953d3d3d";
    const PNG_GREYA16_DIMS: (usize, usize) = (3, 1);
    const PNG_PAL1: &str = "89504e470d0a1a0a0000000d494844520000000a0000000201030000005bafdf930000000e74455874436f6d6d656e74\
         00766563746f72236665970000000467414d410000b18f0bfc610500000006504c544583f85c9e57f47c7b1c53000000\
         074944415478da63787380d1c98d53cf0000000749444154e6080008e102aeabb9b4710000000049454e44ae426082";
    const PNG_PAL1_RGB: &str = "9e57f49e57f49e57f483f85c9e57f49e57f483f85c83f85c9e57f49e57f483f85c83f85c9e57f49e57f49e57f49e57f4\
         83f85c83f85c83f85c83f85c";
    const PNG_PAL1_DIMS: (usize, usize) = (10, 2);
    const PNG_PAL2: &str = "89504e470d0a1a0a0000000d494844520000000700000003020300000022adfd560000000e74455874436f6d6d656e74\
         00766563746f72236665970000000467414d410000b18f0bfc61050000000c504c5445d56db748bd2142235baaa00f75\
         45618a000000084944415478da63086561ac5f9cd5a43a0000000949444154ca14ff0600099202cc133c617400000000\
         49454e44ae426082";
    const PNG_PAL2_RGB: &str = "48bd2148bd2148bd2148bd21d56db7d56db748bd2148bd21aaa00faaa00faaa00fd56db742235b48bd21aaa00f48bd21\
         aaa00f42235bd56db748bd21d56db7";
    const PNG_PAL2_DIMS: (usize, usize) = (7, 3);
    const PNG_PAL4: &str = "89504e470d0a1a0a0000000d494844520000000500000002040300000062440b6e0000000e74455874436f6d6d656e74\
         00766563746f72236665970000000467414d410000b18f0bfc610500000027504c54456691e710fb07dfd8c4d048dc25\
         c42e350b542a02985d9acb0675ad5a9074704c7558159eaa17e0b9cb1c0d000000084944415478da63602b0a60148c9f\
         93230000000849444154dfd8010006a1021a508054c30000000049454e44ae426082";
    const PNG_PAL4_RGB: &str = "6691e72a02985d9acbdfd8c4350b5410fb075d9acbaa17e00675ad350b54";
    const PNG_PAL4_DIMS: (usize, usize) = (5, 2);
    const PNG_PAL8: &str = "89504e470d0a1a0a0000000d49484452000000040000000408030000009e2f6e4c0000000e74455874436f6d6d656e74\
         00766563746f72236665970000000467414d410000b18f0bfc610500000258504c544533a110dadbe7fdcb4738466e56\
         3ec8e68d92e3b5b6f5d94e88c35c5f86bbbb9f54358798929d48ea6e10c6a0d1474302db6b54bc5dff5753fb99133118\
         f6ed72ee6826731ba59b0ceec4287ca22afddf6dfc56fcac40bc0024ec7bc34bba569e663e8704e0372c092cabb5ec9b\
         3c90ad7227c4da966e6b2c5c54ba08499233d15e84e74ef8f8c949e73a2bbfff44c540b623bc76f929500a39a56bbc29\
         9072e1447fd0f33ea77a0232f21701a5e26d71d76abd7496fdf6d84b53ba9b7da77deae7267b39c73785aec0badc87c1\
         21c67b21e39bed11704549211b7ed003e014850a3545406d1d9fc31329e6c3808b7ca0715a7a4ef545fafae4b085078d\
         8018c707e8ac4874d64d698b30221555ea5932fe9cdc56408b7d8cc9b451991b36f7f7527979184ea9108a47e50eaf16\
         6cc631a02f9669357944749a6f4645cc045904dfbcb5b143ec45cfb4ad7de397b4f8ec01667af87adeb3dd88821fb533\
         d044b9452fce69f97198e3c4cc52972a0b2baa17db6ecdaaea18f09cc20ac25d53368e86e7aecb18001ded3bbfac445a\
         5946b3c013cb44d4a70e0f60e9d1bfe9d6f573de64fa33d8728a764ab0ee3a8d12f9a22f204a59e556ef12274889c316\
         3e5369374876cf9ca45ce9686843a06817c1f2cfd51f1590b696b13a703070582462bc545a471defb1249907d3303536\
         5668cce2090c117f4f228acaedf3f97450022799e5e4de224cf5abc9f1fcb497586138440f8f04831d868cc875eac3ae\
         ecec66103d2c46a3c9a48cce1039b38e6e57d38408b2c3a293506cafdc6716528faaeb947fe6c9fbc638d4e4019b1e7e\
         ea49619257c67fe62ff5a460ec1db90000000e4944415478da63f059619dc0a83f6d473b53c158e2910000000e494441\
         5466f1365966e64b7f4c0145e2074fd5ae40820000000049454e44ae426082";
    const PNG_PAL8_RGB: &str = "6d1d9fcaedf36abd7452797944c540619257f09cc2563ec868684332f217a56bbc04e0378b7ca072ee68663e87c707e8";
    const PNG_PAL8_DIMS: (usize, usize) = (4, 4);
    const PNG_ADAM7_RGB8: &str = "89504e470d0a1a0a0000000d494844520000000b0000000908020000011c0171ec0000000e74455874436f6d6d656e74\
         00766563746f72236665970000000467414d410000b18f0bfc6105000000a34944415478da013c01c3fe00af42bea2bf\
         12013db0c3297c680044a62c0105573e00c869ce9ce3036f93b400cc6ce3ec2ef550c4be014fb90d56da33d896dd02e0\
         71cd463839086b8300047cf1ec32e58a12b56045e5684d6ffc2614017cb9e66a6f86820f7a5e13b8e15c9bc4de1e0099\
         c1172f06eeedd30136d8f2526d5301737fdacf3b49b0c890e93a5fd7d4a302210a01e23d2fd5c95cfffa31300d0403f0\
         8145295b2d57a995ff38000000a449444154fae5e09b8a091b77042c5ca1523d3e6556de629d66f99376006aa9c8d653\
         acb1627a7c519f7b9d5239b2acfc704354170e9b9bb9db32bef8c0250128e269182c0d8f994d07805645e27f8dad3ae6\
         fdd82ebca285dd11be2ffc09477902e16ceedff3958cee2c3d101d55ddbf3eb2f78a2bed406eb85eee3311ed30452c2f\
         037184061003d24c7b8a6aa6bbbbe84142de5511be8512151cd6d7e3d4350d07c1b2b62d92b6492365d5000000004945\
         4e44ae426082";
    const PNG_ADAM7_RGB8_RGB: &str = "af42be99c117cc6ce32f06ee44a62cedd301ec2ef536d8f2a2bf12526d5350c4be6aa9c8d653acb1627a7c519f7b9d52\
         39b2acfc704354170e9b9bb9db32bef8c025047cf1737fdaec32e542ba238a12b5f282b36045e5dbbc12684d6fb290b5\
         fc261428e269400e76cfa7c3d627191b0998a8b6d28eb3aabc6f4c414c5dff7b5908c2d2c869ce9489db4fb90d24f752\
         9ce303c74b0fa59340dab6436f93b4e29db97d291d094e571f010b5b95ef13373670e657e668c918de97fcdd049f3a90\
         1068894dee017cb9e63ac5b2e6286c5839af6837e6e63c44c64a9ec014cda7a639da733a6b845775ab315a59f0a6f279\
         c63a12567875e04ef48d544ad6ad43904a4c248e773f7fee3db0c36621532f2adab85e9105573e4bb422ebcb79ad5133\
         662c2bb906a98594a0";
    const PNG_ADAM7_RGB8_DIMS: (usize, usize) = (11, 9);
    const PNG_ADAM7_GREY1: &str = "89504e470d0a1a0a0000000d494844520000000d0000000a01000000013092d9ff0000000e74455874436f6d6d656e74\
         00766563746f72236665970000000467414d410000b18f0bfc61050000001a4944415478da6370603cc0c0c0d8c07080\
         e1006303d3020619c6028612c68388545b0000001b494441540ca61ce6381613866e0e46850ea6a417cc4ee62c061f00\
         e6cf0abf20daa6ea0000000049454e44ae426082";
    const PNG_ADAM7_GREY1_RGB: &str = "000000000000ffffffffffff000000ffffffffffffffffffffffff000000000000ffffff000000ffffff000000000000\
         000000ffffff000000ffffffffffff000000000000000000000000ffffff000000000000000000ffffff000000ffffff\
         ffffff000000ffffffffffffffffff000000000000000000000000ffffff000000000000000000000000000000ffffff\
         000000ffffff000000ffffffffffffffffffffffffffffffffffff000000000000ffffff000000000000000000ffffff\
         000000ffffff000000000000000000000000000000ffffff000000ffffff000000000000ffffff000000000000ffffff\
         ffffffffffffffffff000000ffffff000000000000ffffff000000000000000000ffffff000000000000000000000000\
         000000ffffffffffffffffffffffff000000000000000000ffffffffffff000000ffffffffffffffffff000000ffffff\
         ffffffffffffffffffffffff000000ffffff000000ffffffffffff000000000000ffffffffffffffffff000000ffffff\
         ffffff000000";
    const PNG_ADAM7_GREY1_DIMS: (usize, usize) = (13, 10);
    const PNG_ADAM7_PAL4: &str = "89504e470d0a1a0a0000000d494844520000000300000005040300000105587b070000000e74455874436f6d6d656e74\
         00766563746f72236665970000000467414d410000b18f0bfc61050000001b504c54451b4dc7989ace856e23baba1121\
         8eee367e5c02672861b1d06bdafb409b24330000000f4944415478da63086048607060646050622860089b75ab000000\
         0f494441542c6012600870602ce901002121039894a65c5a0000000049454e44ae426082";
    const PNG_ADAM7_PAL4_RGB: &str = "367e5c61b1d0218eee367e5c1b4dc7218eee856e2361b1d0856e2361b1d0218eee1b4dc70267286bdafb1b4dc7";
    const PNG_ADAM7_PAL4_DIMS: (usize, usize) = (3, 5);
    const PNG_ADAM7_TINY: &str = "89504e470d0a1a0a0000000d49484452000000010000000110060000013882285c0000000e74455874436f6d6d656e74\
         00766563746f72236665970000000467414d410000b18f0bfc6105000000084944415478da637823527338282335f600\
         00000949444154f0109b0500127a0391f92c72140000000049454e44ae426082";
    const PNG_ADAM7_TINY_RGB: &str = "ec7c51";
    const PNG_ADAM7_TINY_DIMS: (usize, usize) = (1, 1);
    const JPEG_BASELINE: &str = "ffd8ffe000104a46494600010100000100010000ffdb0043000201010101010201010102020202020403020202020504\
         040304060506060605060606070908060709070606080b08090a0a0a0a0a06080b0c0b0a0c090a0a0affdb0043010202\
         02020202050303050a0706070a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a\
         0a0a0a0a0a0a0a0a0a0a0a0a0a0affc00011080010001003012200021101031101ffc4001f0000010501010101010100\
         000000000000000102030405060708090a0bffc400b5100002010303020403050504040000017d010203000411051221\
         31410613516107227114328191a1082342b1c11552d1f02433627282090a161718191a25262728292a3435363738393a\
         434445464748494a535455565758595a636465666768696a737475767778797a838485868788898a9293949596979899\
         9aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1\
         f2f3f4f5f6f7f8f9faffc4001f0100030101010101010101010000000000000102030405060708090a0bffc400b51100\
         020102040403040705040400010277000102031104052131061241510761711322328108144291a1b1c109233352f015\
         6272d10a162434e125f11718191a262728292a35363738393a434445464748494a535455565758595a63646566676869\
         6a737475767778797a82838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4\
         c5c6c7c8c9cad2d3d4d5d6d7d8d9dae2e3e4e5e6e7e8e9eaf2f3f4f5f6f7f8f9faffda000c03010002110311003f00f8\
         b7e1c7eca9feaffe25be9fc15ef9f0e3f654ff0057ff0012df4fe0afac7e1cfeca9feaff00e25be9fc15ef7f0e3f654f\
         f57ff12df4fe0afe96f1ebc61fe2fef3bf53c7fa30f8f5fc0fdef6ea7fffd9";
    const JPEG_BASELINE_PIL_RGB: &str = "0101f50b04f21b03eb2b02e23c03db4c02d35b03cb6b04c37c02bc8c03b39b04aba903a3bc039cca0392dc028ce7068a\
         060df91011f91f10ef2f10e7410fe05010d8600fd07010c9800fbf9210b8a010aeaf10a8c0109fd01099e00f8fea128d\
         061efa1021f72021ef3020e54120df5020d66020cf7021c88020bf9120b6a021b0b020a6c1219fd11f97e0208feb238e\
         052ef80f30f71f30ef2f2fe54130e0502fd65f2fd06f30c77f30bd9030b59f31aeaf2fa6c0309fd02f96df308fea318c\
         053ffa1041f62041f02f40e64240df5040d76040d17040c88140be9041b69f41afaf40a6c141a0d04097e1408eea428d\
         054df91051f72050f0304fe74150df514fd7604fd16f50c87f4fbd9150b6a050afae50a6c150a0cf5097e0508eeb518d\
         065dfa1160f9215ff0315fe8425fe1515fd9615ed17160ca815fbe9161b7a05fafb060a9c25fa0d16098e15e90eb628e\
         066dfa1070f82170ef316fe84270e0506fd9616fd07070c8806fbf9071b7a170b1b06fa7c270a0d06f98e16f90ea728e\
         067ef81081f72080ee2f81e74081df5181d6607fcf7081c78080be8f81b6a081adae80a5c0819ecf8096e0808eeb838c\
         068ff91091f62091ef2e91e64190dd5090d76090d07091c77f90bc8f92b59f90adae91a5bf919ecf9196df908cea938c\
         059ff911a1f821a0ef30a1e742a0e051a0d860a0d071a1c981a0bd90a1b59fa1aeafa0a7c0a19fd0a096e19f8feba38d\
         06aef90fb0f820b0ef31b0e741afe051aed760afd06fafc880b0be91b0b5a0b0afaeafa7c1b0a0d1af96e0af8febb18c\
         06befa11c0f720c0f031c0e841c0e051bfd860bfd171bfc982c0bf91c0b6a0c1b0b0c0a6c1c0a1d1bf97e1bf8febc28e\
         07cef810d0f71fd1ef2fcfe741d0e050cfd65fcfd070cfc780cebe90d0b6a1d0b0afcfa6c0d0a1d0cf97e1cf8fead28c\
         07defa10e1f721e1ee31dfe641dfde50e0d760e1cf70e0c780dfbd90e1b6a0e0aeafe1a6c1df9fd2df97e0df8deae28d\
         0aebfd14eefa26edf434ecea47ede355ecdb64edd374edca85ebc395edbba4eeb3b4ecabc5eda4d5ec9ae6ec94efee91";
    const JPEG_PROGRESSIVE: &str = "ffd8ffe000104a46494600010100000100010000ffdb0043000201010101010201010102020202020403020202020504\
         040304060506060605060606070908060709070606080b08090a0a0a0a0a06080b0c0b0a0c090a0a0affdb0043010202\
         02020202050303050a0706070a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a\
         0a0a0a0a0a0a0a0a0a0a0a0a0a0affc20011080010001003012200021101031101ffc400150001010000000000000000\
         0000000000000607ffc4001501010100000000000000000000000000000407ffda000c030100021003100000018b3eac\
         3ea58fffc400161000030000000000000000000000000000000506ffda00080101000105025d2a2e95174a8ba54fffc4\
         001811000203000000000000000000000000000006072232ffda0008010301013f018c1f7163ffc40017110100030000\
         000000000000000000000006002232ffda0008010201013f017ac3569fffc40016100003000000000000000000000000\
         0000000123ffda0008010100063f025314c5314cffc40015100101000000000000000000000000000000f1ffda000801\
         0100013f219a9a9a9affda000c0301000200030000001023ffc4001611000300000000000000000000000000000041a1\
         ffda0008010301013f1086cfffc4001611000300000000000000000000000000000041a1ffda0008010201013f10accf\
         ffc40015100101000000000000000000000000000000a1ffda0008010100013f1098cc62331fffd9";
    const JPEG_PROGRESSIVE_PIL_RGB: &str = "0101f50b04f21b03eb2b02e23c03db4c02d35b03cb6b04c37c02bc8c03b39b04aba903a3bc039cca0392dc028ce7068a\
         060df91011f91f10ef2f10e7410fe05010d8600fd07010c9800fbf9210b8a010aeaf10a8c0109fd01099e00f8fea128d\
         061efa1021f72021ef3020e54120df5020d66020cf7021c88020bf9120b6a021b0b020a6c1219fd11f97e0208feb238e\
         052ef80f30f71f30ef2f2fe54130e0502fd65f2fd06f30c77f30bd9030b59f31aeaf2fa6c0309fd02f96df308fea318c\
         053ffa1041f62041f02f40e64240df5040d76040d17040c88140be9041b69f41afaf40a6c141a0d04097e1408eea428d\
         054df91051f72050f0304fe74150df514fd7604fd16f50c87f4fbd9150b6a050afae50a6c150a0cf5097e0508eeb518d\
         065dfa1160f9215ff0315fe8425fe1515fd9615ed17160ca815fbe9161b7a05fafb060a9c25fa0d16098e15e90eb628e\
         066dfa1070f82170ef316fe84270e0506fd9616fd07070c8806fbf9071b7a170b1b06fa7c270a0d06f98e16f90ea728e\
         067ef81081f72080ee2f81e74081df5181d6607fcf7081c78080be8f81b6a081adae80a5c0819ecf8096e0808eeb838c\
         068ff91091f62091ef2e91e64190dd5090d76090d07091c77f90bc8f92b59f90adae91a5bf919ecf9196df908cea938c\
         059ff911a1f821a0ef30a1e742a0e051a0d860a0d071a1c981a0bd90a1b59fa1aeafa0a7c0a19fd0a096e19f8feba38d\
         06aef90fb0f820b0ef31b0e741afe051aed760afd06fafc880b0be91b0b5a0b0afaeafa7c1b0a0d1af96e0af8febb18c\
         06befa11c0f720c0f031c0e841c0e051bfd860bfd171bfc982c0bf91c0b6a0c1b0b0c0a6c1c0a1d1bf97e1bf8febc28e\
         07cef810d0f71fd1ef2fcfe741d0e050cfd65fcfd070cfc780cebe90d0b6a1d0b0afcfa6c0d0a1d0cf97e1cf8fead28c\
         07defa10e1f721e1ee31dfe641dfde50e0d760e1cf70e0c780dfbd90e1b6a0e0aeafe1a6c1df9fd2df97e0df8deae28d\
         0aebfd14eefa26edf434ecea47ede355ecdb64edd374edca85ebc395edbba4eeb3b4ecabc5eda4d5ec9ae6ec94efee91";
    const JPEG_444: &str = "ffd8ffe000104a46494600010100000100010000ffdb0043000201010101010201010102020202020403020202020504\
         040304060506060605060606070908060709070606080b08090a0a0a0a0a06080b0c0b0a0c090a0a0affdb0043010202\
         02020202050303050a0706070a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a\
         0a0a0a0a0a0a0a0a0a0a0a0a0a0affc00011080010001003011100021101031101ffc4001f0000010501010101010100\
         000000000000000102030405060708090a0bffc400b5100002010303020403050504040000017d010203000411051221\
         31410613516107227114328191a1082342b1c11552d1f02433627282090a161718191a25262728292a3435363738393a\
         434445464748494a535455565758595a636465666768696a737475767778797a838485868788898a9293949596979899\
         9aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1\
         f2f3f4f5f6f7f8f9faffc4001f0100030101010101010101010000000000000102030405060708090a0bffc400b51100\
         020102040403040705040400010277000102031104052131061241510761711322328108144291a1b1c109233352f015\
         6272d10a162434e125f11718191a262728292a35363738393a434445464748494a535455565758595a63646566676869\
         6a737475767778797a82838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4\
         c5c6c7c8c9cad2d3d4d5d6d7d8d9dae2e3e4e5e6e7e8e9eaf2f3f4f5f6f7f8f9faffda000c03010002110311003f00f8\
         b7e1c7eca9feaffe25be9fc15feab71ef187c7ef1f0fe1871efc1eff0063df3e1c7eca9feaff00e25be9fc15fc53c7bc\
         61f1fbc7fa2be1871efc1ef9f58fc39fd953fd5ffc4b7d3f82beb38f78c3e3f78ff053c30e3df83df3defe1c7eca9fea\
         ff00e25be9fc15fc53c7bc61f1fbc7fa2be1871efc1ef9ffd9";
    const JPEG_444_PIL_RGB: &str = "0000fe1000f71f01ed2e01e44200e05000d75e01cc7101c78100bd8f01b8a100aeb000a5c0019fd00098e1008def0288\
         000fff1210fa2010ef2f10e7430ee2520fd95f10ce7210c9810fbf9011baa10faeb20fa8c10fa1d20f99e10f8ef01188\
         0021ff1020f91f21ef2d21e54120df5020d85d21ce7021c88020bf8e21b8a021aeb020a4c021a1cf2097e0208def2287\
         0031fe1030f71e31ed2c31e54130e04f30d65e31cc6f30c57f30bf8d31b89f31aeaf30a5bf31a1cf3097e12f8dee3086\
         0140ff123ffa2140f02f40e6443fe1523fd96040cf723fc8823fc09040b9a240afb23fa6c240a2d04098e23f8ef04088\
         004ffe124ff82050ee2e50e7434fe1514fd76050cd714fc7804ebf8f50b9a150afb14fa6c150a2d14f99e34e8eef5088\
         0061fd1161f81e61ed2d62e44160df5061d75e61cc7061c67f60be8e62b99f61aeb060a5bf619fd06098e15f8dee6287\
         006fff126ffa1f71ef2e71e6426fe2506fd96070cf7070c8806fbf8f71b9a171afb06fa7c071a1d06f98e26f8ef07089\
         0080fd0f82f92080ee3180e54081df5180d8627fcd6f82c78080be8e81b7a081adaf80a4c081a0cf8096e17f8cf08187\
         0290ff1091f92290ef318fe7418fe1518fd8638fce7190c7828ec08f91baa08fafb18fa7c08fa2d18f99e28e8ef19089\
         00a0fe0ea2f81fa1ed30a1e53fa1e050a1d861a0cc6ea2c781a0bf8da1b99fa1aeafa0a5bfa19fcfa098e1a08eefa288\
         00affe0eb0f81fb1ed2fb1e53eb1e04eb0d760afcd6eb0c680b0be8eb1b7a0b0adaeb0a5beb1a0ceb096e0b08ceeb185\
         02bfff10c0f922c0ef32c0e641c0e151bfd862bfce70c0c782bfc08fc0baa1c0b0b1bfa6c1c0a1d1bf99e3bf8ff1c088\
         02ceff0fd0f921d0ef30cfe541d0e150cfd862cece70cfc781cebe90d0b8a1d0aeb0cfa6c0d0a1d0cf97e2cf8df0d087\
         00e1ff0ee2f820e2ee2fe1e53ee1de4fe1d761e0cd6fe1c780dfbd8fe2b89ee1acafe1a4bee19fcfe197e0df8cefe286\
         02efff0ef1f820f1ee30efe740f1e150f0d862efcd6ff0c980eebf8ff0baa0f1afb0f0a6c0f1a2cff099e1ef8eeff088";
    const JPEG_GREY: &str = "ffd8ffe000104a46494600010100000100010000ffdb0043000201010101010201010102020202020403020202020504\
         040304060506060605060606070908060709070606080b08090a0a0a0a0a06080b0c0b0a0c090a0a0affc0000b080010\
         001001011100ffc4001f0000010501010101010100000000000000000102030405060708090a0bffc400b51000020103\
         03020403050504040000017d01020300041105122131410613516107227114328191a1082342b1c11552d1f024336272\
         82090a161718191a25262728292a3435363738393a434445464748494a535455565758595a636465666768696a737475\
         767778797a838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9\
         cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9faffda0008010100003f00f8b7e1c7eca9feaf\
         fe25be9fc15ef9f0e3f654ff0057ff0012df4fe0afac7e1cfeca9feaff00e25be9fc15ef7f0e3f654ff57ff12df4fe0a\
         ffd9";
    const JPEG_GREY_PIL_RGB: &str = "1d1d1d2121212525252828282d2d2d3030303434343939393c3c3c4040404444444747474c4c4c4f4f4f535353585858\
         2626262b2b2b2e2e2e3232323636363a3a3a3d3d3d4242424545454a4a4a4d4d4d5151515555555959595c5c5c616161\
         3030303434343838383b3b3b4040404343434747474c4c4c4f4f4f5353535757575a5a5a5f5f5f6262626666666b6b6b\
         3939393d3d3d4141414444444949494c4c4c5050505454545858585c5c5c6060606363636868686b6b6b6f6f6f737373\
         4343434747474b4b4b4e4e4e5353535656565a5a5a5e5e5e6262626666666a6a6a6d6d6d7272727575757979797d7d7d\
         4b4b4b5050505454545757575c5c5c5f5f5f6363636767676a6a6a6f6f6f7373737676767b7b7b7e7e7e828282868686\
         5555555a5a5a5d5d5d6161616565656969696c6c6c7171717474747979797c7c7c8080808484848888888b8b8b909090\
         5e5e5e6363636767676a6a6a6f6f6f7272727676767a7a7a7d7d7d8282828686868989898e8e8e919191959595999999\
         6868686d6d6d7070707474747878787c7c7c7f7f7f8484848787878b8b8b8f8f8f9292929797979a9a9a9e9e9ea3a3a3\
         7272727676767a7a7a7d7d7d8181818585858989898d8d8d9090909595959898989c9c9ca0a0a0a4a4a4a7a7a7acacac\
         7b7b7b8080808383838787878b8b8b8f8f8f9292929797979a9a9a9e9e9ea2a2a2a5a5a5aaaaaaadadadb1b1b1b6b6b6\
         8484848888888c8c8c9090909494949797979b9b9b9f9f9fa3a3a3a7a7a7abababaeaeaeb3b3b3b6b6b6babababebebe\
         8e8e8e9292929696969a9a9a9e9e9ea1a1a1a5a5a5a9a9a9adadadb1b1b1b5b5b5b8b8b8bdbdbdc0c0c0c4c4c4c8c8c8\
         9797979b9b9b9f9f9fa2a2a2a7a7a7aaaaaaaeaeaeb2b2b2b5b5b5babababebebec1c1c1c6c6c6c9c9c9cdcdcdd1d1d1\
         a1a1a1a5a5a5a9a9a9acacacb0b0b0b4b4b4b8b8b8bcbcbcbfbfbfc4c4c4c7c7c7cbcbcbcfcfcfd3d3d3d6d6d6dbdbdb\
         aaaaaaaeaeaeb2b2b2b5b5b5babababdbdbdc1c1c1c5c5c5c8c8c8cdcdcdd1d1d1d4d4d4d9d9d9dcdcdce0e0e0e4e4e4";
    const JPEG_CMYK: &str = "ffd8ffee000e41646f626500640000000000ffdb00430002010101010102010101020202020204030202020205040403\
         04060506060605060606070908060709070606080b08090a0a0a0a0a06080b0c0b0a0c090a0a0affc000140800100010\
         044311004d11005911004b1100ffc4001f0000010501010101010100000000000000000102030405060708090a0bffc4\
         00b5100002010303020403050504040000017d01020300041105122131410613516107227114328191a1082342b1c115\
         52d1f02433627282090a161718191a25262728292a3435363738393a434445464748494a535455565758595a63646566\
         6768696a737475767778797a838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2\
         c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9faffda000e0443004d005900\
         4b00003f00fcdfff00826dff00cb87fc06bf37ff00e1db7ff500ff00c855fb19fb497fcbc7e35fbf95fd007fc136ff00\
         e5c3fe0347fc3b6ffea01ff90abf3fff00692ff978fc68afe7ff00fe09b7ff002e1ff01afe803fe1db7ff500ff00c855\
         fa01fb497fcbc7e3457f401ff04dbff970ff0080d1ff000edbff00a807fe42afcfff00da4bfe5e3f1a2bffd9";
    const JPEG_CMYK_PIL_RGB: &str = "0000ff1000f72000ef3000e74000df5000d76000cf7000c78000bf9000b7a000afb000a7c0009fd00097e0008ff00087\
         0010ff1010f72010ef3010e74010df5010d76010cf7010c78010bf9010b7a010afb010a7c0109fd01097e0108ff01087\
         0020ff1020f72020ef3020e74020df5020d76020cf7020c78020bf9020b7a020afb020a7c0209fd02097e0208ff02087\
         0030ff1030f72030ef3030e74030df5030d76030cf7030c78030bf9030b7a030afb030a7c0309fd03097e0308ff03087\
         0041ff1041f72041ef3041e74041df5041d76041cf7041c78041bf9041b7a041afb041a7c0419fd04197e0418ff04187\
         0050ff1050f72050ef3050e74050df5050d76050cf7050c78050bf9050b7a050afb050a7c0509fd05097e0508ff05087\
         0060ff1060f72060ef3060e74060df5060d76060cf7060c78060bf9060b7a060afb060a7c0609fd06097e0608ff06087\
         0070ff1070f72070ef3070e74070df5070d76070cf7070c78070bf9070b7a070afb070a7c0709fd07097e0708ff07087\
         0080ff1080f72080ef3080e74080df5080d76080cf7080c78080bf9080b7a080afb080a7c0809fd08097e0808ff08087\
         0090ff1090f72090ef3090e74090df5090d76090cf7090c78090bf9090b7a090afb090a7c0909fd09097e0908ff09087\
         00a0ff10a0f720a0ef30a0e740a0df50a0d760a0cf70a0c780a0bf90a0b7a0a0afb0a0a7c0a09fd0a097e0a08ff0a087\
         00b0ff10b0f720b0ef30b0e740b0df50b0d760b0cf70b0c780b0bf90b0b7a0b0afb0b0a7c0b09fd0b097e0b08ff0b087\
         00c1ff10c1f720c1ef30c1e740c1df50c1d760c1cf70c1c780c1bf90c1b7a0c1afb0c1a7c0c19fd0c197e0c18ff0c187\
         00d0ff10d0f720d0ef30d0e740d0df50d0d760d0cf70d0c780d0bf90d0b7a0d0afb0d0a7c0d09fd0d097e0d08ff0d087\
         00e0ff10e0f720e0ef30e0e740e0df50e0d760e0cf70e0c780e0bf90e0b7a0e0afb0e0a7c0e09fd0e097e0e08ff0e087\
         00f0ff10f0f720f0ef30f0e740f0df50f0d760f0cf70f0c780f0bf90f0b7a0f0afb0f0a7c0f09fd0f097e0f08ff0f087";
}
