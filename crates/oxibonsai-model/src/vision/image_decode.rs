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
//! A caller that knows its per-image token budget (`--image-max-tokens`)
//! bounds the decode tighter still: [`ImageSourcePolicy::max_source_pixels`]
//! is the most pixels a source may have, checked at the same point (the PNG
//! `IHDR`, the JPEG frame header) and refused with
//! [`ImageInputError::OverDecodeBudget`] before a byte is inflated. The
//! budget follows from the token budget
//! ([`super::preprocess::source_pixel_budget`]): an image far larger than
//! the grid it is resized to is not worth the memory of decoding it — a
//! half-megabyte PNG can declare 8192 x 8192 pixels and cost over a gigabyte
//! to inflate, unfilter and convert.
//!
//! # Sources
//!
//! [`load_image_source`] resolves an OpenAI `image_url` / CLI `--image`
//! reference under an [`ImageSourcePolicy`]: base64 `data:` URIs always,
//! local files only where the policy allows them (the CLI names its own
//! files; a server resolves `file://` references inside an operator-chosen
//! media directory only — `--media-path <dir>` (or `OXI_MEDIA_PATH`) for
//! `oxibonsai serve`, as the reference server's `--media-path` does), and
//! `http(s)` only through the remote-image fetcher the policy carries
//! ([`ImageSourcePolicy::remote`], see [`super::remote`]). Fetching
//! arbitrary URLs is a server-side request forgery surface, so by default
//! a remote reference is refused with a typed error before anything is
//! opened or resolved; the operator opts in (`--allow-image-url-fetch` or
//! `OXI_ALLOW_IMAGE_URL_FETCH=1`), and the front end installs a fetcher
//! that applies the address policy of [`super::remote`] (the `oxibonsai`
//! command does). The fetched bytes are decoded exactly like a data URI's:
//! the format is sniffed from the bytes and the decode budget applies.

use std::path::{Component, Path, PathBuf};

use super::remote::{fetch_remote_reference, RemoteImageAccess, SharedRemoteImageFetcher};
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
    /// The image is within the hard limits ([`MAX_DECODED_PIXELS`],
    /// [`MAX_IMAGE_SIDE`]) but has more pixels than the decode budget its
    /// caller derived from the per-image token budget
    /// ([`ImageSourcePolicy::max_source_pixels`]).
    #[error(
        "the image is {width} x {height} pixels, over the decode budget of {max_pixels} pixels \
         that the per-image token budget (--image-max-tokens) allows; downscale the image, or \
         raise --image-max-tokens to admit larger sources"
    )]
    OverDecodeBudget {
        /// Declared width.
        width: usize,
        /// Declared height.
        height: usize,
        /// The budget, in pixels.
        max_pixels: usize,
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
    /// An `http(s)` reference the policy does not fetch at all: the
    /// operator did not opt in, or no fetcher is installed (see
    /// [`super::remote::RemoteImageAccess`]).
    #[error(
        "remote image URLs are not fetched ({url}): {reason}; send the image inline as a base64 \
         data URI (data:image/png;base64,...)"
    )]
    RemoteFetchRefused {
        /// The URL, truncated for the message.
        url: String,
        /// Why (not opted in, or opted in without a fetcher).
        reason: String,
    },
    /// An `http(s)` reference the remote-image address policy refuses: its
    /// syntax, its scheme, credentials in it, or a host that is not a public
    /// address and not allowlisted ([`super::remote::RemoteUrlRefusal`]).
    #[error("remote image URL refused ({url}): {reason}")]
    RemoteUrlRefused {
        /// The URL, truncated for the message.
        url: String,
        /// Which rule refused it (never a resolved address).
        reason: String,
    },
    /// A permitted `http(s)` reference whose fetch failed, naming the step
    /// ([`super::remote::RemoteFetchFailure`]: resolve, connect, TLS, a
    /// status other than 200, a redirect, the body, or the deadline).
    #[error("remote image fetch failed ({url}): {reason}")]
    RemoteFetchFailed {
        /// The URL, truncated for the message.
        url: String,
        /// The step that failed.
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
            Self::EncodedTooLarge { .. }
            | Self::TooLarge { .. }
            | Self::OverDecodeBudget { .. } => "image_too_large",
            Self::Empty { .. } => "image_empty",
            Self::TooManyTokens { .. } => "image_too_many_tokens",
            Self::DataUri { .. } => "image_data_uri_invalid",
            Self::UnsupportedSource { .. } => "image_url_scheme_unsupported",
            Self::RemoteFetchRefused { .. } => "image_url_fetch_disabled",
            Self::RemoteUrlRefused { .. } => "image_url_refused",
            Self::RemoteFetchFailed { .. } => "image_url_fetch_failed",
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

/// Refuse a declared size before any pixel buffer is allocated: a zero side,
/// anything past the hard limits ([`MAX_DECODED_PIXELS`], [`MAX_IMAGE_SIDE`]),
/// and — when the caller gave one — more pixels than `budget`
/// ([`ImageSourcePolicy::max_source_pixels`]).
fn check_dimensions(
    width: usize,
    height: usize,
    budget: Option<usize>,
) -> Result<(), ImageInputError> {
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
    let pixels = match width.checked_mul(height) {
        Some(pixels) if pixels <= MAX_DECODED_PIXELS => pixels,
        _ => return Err(too_large()),
    };
    match budget {
        Some(max_pixels) if pixels > max_pixels => Err(ImageInputError::OverDecodeBudget {
            width,
            height,
            max_pixels,
        }),
        _ => Ok(()),
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
        // Never the bytes themselves: the input may be a resource the server
        // fetched on a client's behalf, and this text goes back to that client.
        None => "not a recognised image format".to_string(),
    }
}

/// Decode a PNG or JPEG into RGB8 pixels the way the reference's
/// `stb_image` does (see the module docs).
///
/// # Errors
///
/// [`ImageInputError`] naming the format and the reason; never a panic.
pub fn decode_image(bytes: &[u8]) -> Result<ImageRgb8, ImageInputError> {
    decode_image_within(bytes, None)
}

/// [`decode_image`] under a pixel budget: an image with more than
/// `max_pixels` pixels (`None`: only the hard limits) is refused with
/// [`ImageInputError::OverDecodeBudget`] as soon as its header is read — for
/// a PNG before any `IDAT` data is collected, for a JPEG before any pixel
/// buffer exists.
///
/// # Errors
///
/// As [`decode_image`], plus [`ImageInputError::OverDecodeBudget`].
pub fn decode_image_within(
    bytes: &[u8],
    max_pixels: Option<usize>,
) -> Result<ImageRgb8, ImageInputError> {
    if bytes.len() > MAX_ENCODED_IMAGE_BYTES {
        return Err(ImageInputError::EncodedTooLarge {
            bytes: bytes.len(),
            limit: MAX_ENCODED_IMAGE_BYTES,
        });
    }
    match sniff_format(bytes) {
        Some(ImageFormat::Png) => decode_png_within(bytes, max_pixels),
        Some(ImageFormat::Jpeg) => decode_jpeg_within(bytes, max_pixels),
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
    fn parse(data: &[u8], budget: Option<usize>) -> Result<Self, ImageInputError> {
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
        check_dimensions(width, height, budget)?;
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
    decode_png_within(bytes, None)
}

/// [`decode_png`] under a pixel budget, checked at the `IHDR` — before the
/// `IDAT` data is collected, let alone inflated (see [`decode_image_within`]).
///
/// # Errors
///
/// As [`decode_png`], plus [`ImageInputError::OverDecodeBudget`].
pub fn decode_png_within(
    bytes: &[u8],
    max_pixels: Option<usize>,
) -> Result<ImageRgb8, ImageInputError> {
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
                header = Some(PngHeader::parse(data, max_pixels)?);
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
    decode_jpeg_within(bytes, None)
}

/// [`decode_jpeg`] under a pixel budget, checked at the frame header (see
/// [`decode_image_within`]).
///
/// # Errors
///
/// As [`decode_jpeg`], plus [`ImageInputError::OverDecodeBudget`].
pub fn decode_jpeg_within(
    bytes: &[u8],
    max_pixels: Option<usize>,
) -> Result<ImageRgb8, ImageInputError> {
    // A third-party decoder on caller bytes: a panic inside it must become
    // a typed error, never take the process down.
    let result = std::panic::catch_unwind(|| decode_jpeg_inner(bytes, max_pixels));
    result.unwrap_or_else(|_| {
        Err(ImageInputError::Malformed {
            format: "JPEG",
            reason: "the decoder rejected the data".to_string(),
        })
    })
}

fn decode_jpeg_inner(
    bytes: &[u8],
    max_pixels: Option<usize>,
) -> Result<ImageRgb8, ImageInputError> {
    let mut decoder = jpeg_decoder::Decoder::new(bytes);
    decoder.read_info().map_err(jpeg_error)?;
    let info = decoder.info().ok_or_else(|| ImageInputError::Malformed {
        format: "JPEG",
        reason: "no frame header".to_string(),
    })?;
    let (width, height) = (usize::from(info.width), usize::from(info.height));
    check_dimensions(width, height, max_pixels)?;
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
    /// A directory `file://` references resolve inside (`--media-path <dir>`
    /// or `OXI_MEDIA_PATH` for `oxibonsai serve`, like the reference
    /// server's `--media-path`): no absolute path, no `..`, and the
    /// resolved file must stay inside it once symlinks are followed
    /// (anything else is [`ImageInputError::LocalFileRefused`]). `None` with
    /// `allow_any_local_path == false` refuses local files — a network
    /// server's default.
    pub media_root: Option<PathBuf>,
    /// How remote `http(s)` references are treated (see
    /// [`super::remote`], "Three states"): refused before anything is
    /// opened ([`RemoteImageAccess::Disabled`], the default), refused saying
    /// the operator's opt-in (`--allow-image-url-fetch` or
    /// `OXI_ALLOW_IMAGE_URL_FETCH=1`) has no fetcher to act on
    /// ([`RemoteImageAccess::OptedInWithoutFetcher`]), or fetched through the
    /// installed fetcher ([`RemoteImageAccess::Fetcher`],
    /// [`ImageSourcePolicy::with_remote_fetcher`]).
    pub remote: RemoteImageAccess,
    /// The most pixels a source image may have to be decoded at all: more is
    /// [`ImageInputError::OverDecodeBudget`], decided from the header before
    /// any pixel memory is spent. `None` (the default) leaves only the hard
    /// limits ([`MAX_DECODED_PIXELS`], [`MAX_IMAGE_SIDE`]); a caller that
    /// knows its per-image token budget sets the budget that follows from it
    /// with [`ImageSourcePolicy::with_token_budget`].
    pub max_source_pixels: Option<usize>,
}

impl ImageSourcePolicy {
    /// The policy for a command-line user's own references: local files
    /// allowed, no remote fetch.
    #[must_use]
    pub fn local_user() -> Self {
        Self {
            allow_any_local_path: true,
            media_root: None,
            remote: RemoteImageAccess::Disabled,
            max_source_pixels: None,
        }
    }

    /// The policy for a network server: `data:` URIs only, unless the
    /// operator named a media directory (`file://` references inside it).
    /// `remote_opt_in` records the operator's opt-in to remote references;
    /// they are fetched only once a fetcher is installed
    /// ([`ImageSourcePolicy::with_remote_fetcher`]).
    #[must_use]
    pub fn server(media_root: Option<PathBuf>, remote_opt_in: bool) -> Self {
        Self {
            allow_any_local_path: false,
            media_root,
            remote: if remote_opt_in {
                RemoteImageAccess::OptedInWithoutFetcher
            } else {
                RemoteImageAccess::Disabled
            },
            max_source_pixels: None,
        }
    }

    /// This policy fetching remote references through `fetcher` — the
    /// operator's opt-in made effective by a front end that applies an
    /// address policy (see [`super::remote`]).
    #[must_use]
    pub fn with_remote_fetcher(mut self, fetcher: SharedRemoteImageFetcher) -> Self {
        self.remote = RemoteImageAccess::Fetcher(fetcher);
        self
    }

    /// This policy with its remote access set to `remote`.
    #[must_use]
    pub fn with_remote_access(mut self, remote: RemoteImageAccess) -> Self {
        self.remote = remote;
        self
    }

    /// This policy with its decode budget set to what a per-image token
    /// budget of `max_tokens` (`--image-max-tokens`) justifies for a Qwen-VL
    /// projector: [`super::preprocess::source_pixel_budget`] at the tower's
    /// 32-pixel merge unit. A source image larger than that is refused from
    /// its header, before it is inflated or converted.
    #[must_use]
    pub fn with_token_budget(mut self, max_tokens: usize) -> Self {
        self.max_source_pixels = Some(super::preprocess::source_pixel_budget(
            max_tokens,
            super::preprocess::QWEN_VL_MERGE_UNIT,
        ));
        self
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
///
/// The size that is checked is the size of the file that is read: the path
/// is opened once, its length is taken from the open handle, and the read
/// goes through that handle capped at one byte past the limit — so a file
/// replaced or grown after the check (or one whose length lies, like a
/// pipe's) can never make this buffer more than the limit.
///
/// A path that is not a regular file (a FIFO would block the `open`, a
/// device never ends) is refused by a plain `metadata` check before it is
/// opened; the handle's own metadata repeats the check for the file that was
/// actually opened.
fn read_limited(path: &Path, shown: &str) -> Result<Vec<u8>, ImageInputError> {
    let unreadable = |reason: String| ImageInputError::FileUnreadable {
        path: shown.to_string(),
        reason,
    };
    let not_regular = || unreadable("not a regular file".to_string());
    let on_path = std::fs::metadata(path).map_err(|e| unreadable(e.to_string()))?;
    if !on_path.is_file() {
        return Err(not_regular());
    }
    let mut file = std::fs::File::open(path).map_err(|e| unreadable(e.to_string()))?;
    let on_handle = file.metadata().map_err(|e| unreadable(e.to_string()))?;
    if !on_handle.is_file() {
        return Err(not_regular());
    }
    read_bounded(&mut file, on_handle.len(), MAX_ENCODED_IMAGE_BYTES).map_err(|error| match error {
        BoundedReadError::TooLarge(bytes) => ImageInputError::EncodedTooLarge {
            bytes,
            limit: MAX_ENCODED_IMAGE_BYTES,
        },
        BoundedReadError::Io(reason) => unreadable(reason),
    })
}

/// Why [`read_bounded`] gave up.
#[derive(Debug, PartialEq, Eq)]
enum BoundedReadError {
    /// More than the limit: the byte count seen (the declared length, or one
    /// past the limit when the reader outgrew it).
    TooLarge(usize),
    /// The reader failed.
    Io(String),
}

/// Read all of `reader`, which declares `declared_len` bytes, refusing more
/// than `limit`: a declared length past the limit is refused without reading,
/// and a reader that yields more than the limit anyway (it grew, or its
/// length was never true) is stopped one byte past it by `take`.
fn read_bounded(
    reader: impl std::io::Read,
    declared_len: u64,
    limit: usize,
) -> Result<Vec<u8>, BoundedReadError> {
    use std::io::Read as _;
    let declared = usize::try_from(declared_len).unwrap_or(usize::MAX);
    if declared > limit {
        return Err(BoundedReadError::TooLarge(declared));
    }
    let mut bytes = Vec::with_capacity(declared);
    let cap = u64::try_from(limit).unwrap_or(u64::MAX).saturating_add(1);
    reader
        .take(cap)
        .read_to_end(&mut bytes)
        .map_err(|e| BoundedReadError::Io(e.to_string()))?;
    if bytes.len() > limit {
        return Err(BoundedReadError::TooLarge(bytes.len()));
    }
    Ok(bytes)
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
/// module docs), without decoding it. A remote reference is fetched only
/// through the policy's installed fetcher, held to
/// [`MAX_ENCODED_IMAGE_BYTES`].
///
/// # Errors
///
/// [`ImageInputError`]: a malformed data URI, a refused or unreadable local
/// file, a remote URL that is not fetched (`image_url_fetch_disabled`), is
/// refused by the address policy (`image_url_refused`) or failed to fetch
/// (`image_url_fetch_failed`), an unsupported scheme or an oversized input.
pub fn load_image_bytes(
    reference: &str,
    policy: &ImageSourcePolicy,
) -> Result<Vec<u8>, ImageInputError> {
    let reference = reference.trim();
    match classify_image_source(reference)? {
        ImageSource::DataUri => parse_data_uri(reference).map(|uri| uri.bytes),
        ImageSource::Remote => {
            fetch_remote_reference(reference, &policy.remote, MAX_ENCODED_IMAGE_BYTES)
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
                         directory (--media-path <dir> or OXI_MEDIA_PATH); send the image as a \
                         base64 data URI"
                    ),
                }),
            }
        }
    }
}

/// [`load_image_bytes`] followed by [`decode_image_within`] under the
/// policy's decode budget ([`ImageSourcePolicy::max_source_pixels`]).
///
/// # Errors
///
/// As the two calls it makes.
pub fn load_image_source(
    reference: &str,
    policy: &ImageSourcePolicy,
) -> Result<ImageRgb8, ImageInputError> {
    decode_image_within(
        &load_image_bytes(reference, policy)?,
        policy.max_source_pixels,
    )
}

#[cfg(test)]
#[path = "image_decode_tests.rs"]
mod tests;
