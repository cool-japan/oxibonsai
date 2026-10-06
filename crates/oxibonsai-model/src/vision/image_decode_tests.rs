//! Unit tests for `image_decode.rs` (a sibling file, declared there with
//! `#[path]`, so `super` still names that module).

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

/// A PNG holding only a header and an end marker: enough for every size
/// check, which runs on the header alone.
fn png_header_only(width: u32, height: u32, color_type: u8, bit_depth: u8) -> Vec<u8> {
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[bit_depth, color_type, 0, 0, 0]);
    let mut png = PNG_SIGNATURE.to_vec();
    png.extend_from_slice(&13u32.to_be_bytes());
    png.extend_from_slice(b"IHDR");
    png.extend_from_slice(&ihdr);
    png.extend_from_slice(&crc32(&[b"IHDR", &ihdr]).to_be_bytes());
    png.extend_from_slice(&0u32.to_be_bytes());
    png.extend_from_slice(b"IEND");
    png.extend_from_slice(&crc32(&[b"IEND"]).to_be_bytes());
    png
}

/// The decode budget is decided at the `IHDR`, before any `IDAT` is looked
/// at: a header declaring 8192 x 8192 pixels (a half-megabyte PNG can, and
/// would need over a gigabyte to inflate) is `image_too_large` under a
/// 16 Mpx budget, while without one the same bytes get as far as the
/// missing image data.
#[test]
fn a_decode_budget_refuses_a_large_png_header_before_any_image_data() {
    let big = png_header_only(8192, 8192, 6, 16);
    let budget = 16 * 1024 * 1024;

    let err = decode_image_within(&big, Some(budget)).expect_err("over the budget");
    assert_eq!(err.code(), "image_too_large", "{err}");
    assert_eq!(
        err,
        ImageInputError::OverDecodeBudget {
            width: 8192,
            height: 8192,
            max_pixels: budget,
        }
    );
    let message = err.to_string();
    assert!(message.contains("8192 x 8192"), "{message}");
    assert!(message.contains(&budget.to_string()), "{message}");
    assert!(message.contains("--image-max-tokens"), "{message}");

    // Without a budget only the hard limits apply: the header is fine and
    // the (absent) image data is what fails — proof that the refusal above
    // came from the budget, at the header.
    let err = decode_image(&big).expect_err("no image data");
    assert_eq!(err.code(), "image_decode_failed", "{err}");
    assert!(err.to_string().contains("no IDAT"), "{err}");
}

/// A 12-megapixel phone photo passes the default budget's gate; the budget
/// is a ceiling on pixels, not on bytes or sides.
#[test]
fn a_decode_budget_admits_what_fits_and_names_what_does_not() {
    let photo = png_header_only(4032, 3024, 2, 8);
    let default_budget = super::super::preprocess::source_pixel_budget(
        super::super::preprocess::DEFAULT_IMAGE_MAX_TOKENS,
        super::super::preprocess::QWEN_VL_MERGE_UNIT,
    );
    let err = decode_image_within(&photo, Some(default_budget)).expect_err("no image data");
    assert_eq!(
        err.code(),
        "image_decode_failed",
        "past the budget gate, failing only on the missing IDAT: {err}"
    );

    // Exactly the budget passes the gate; one pixel more does not.
    let exact = png_header_only(1000, 100, 2, 8);
    assert_eq!(
        decode_image_within(&exact, Some(100_000))
            .expect_err("no image data")
            .code(),
        "image_decode_failed"
    );
    let over = decode_image_within(&exact, Some(99_999)).expect_err("one pixel over");
    assert_eq!(over.code(), "image_too_large", "{over}");

    // The hard limits stay the hard limits, named as before, whatever the
    // budget says; a zero side is still empty.
    let huge = png_header_only(40_000, 40_000, 2, 8);
    let err = decode_image_within(&huge, Some(usize::MAX)).expect_err("hard limit");
    assert!(matches!(err, ImageInputError::TooLarge { .. }), "{err}");
    let empty = png_header_only(0, 4, 2, 8);
    assert_eq!(
        decode_image_within(&empty, Some(1))
            .expect_err("empty")
            .code(),
        "image_empty"
    );
}

/// Both formats honour the budget at their header, on real encodings: the
/// 5 x 4 PNG and the 16 x 16 baseline and progressive JPEGs decode at
/// exactly their pixel count and are refused one pixel below it.
#[test]
fn both_formats_honour_a_pixel_budget_at_their_header() {
    for (name, bytes, pixels) in [
        ("png", hex(PNG_RGB8), 5 * 4),
        ("baseline jpeg", hex(JPEG_BASELINE), 16 * 16),
        ("progressive jpeg", hex(JPEG_PROGRESSIVE), 16 * 16),
        ("grey jpeg", hex(JPEG_GREY), 16 * 16),
    ] {
        let image =
            decode_image_within(&bytes, Some(pixels)).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(image.width * image.height, pixels, "{name}");
        let err = decode_image_within(&bytes, Some(pixels - 1)).expect_err(name);
        assert_eq!(
            err,
            ImageInputError::OverDecodeBudget {
                width: image.width,
                height: image.height,
                max_pixels: pixels - 1,
            },
            "{name}"
        );
        assert_eq!(err.code(), "image_too_large", "{name}");
        assert_eq!(
            decode_image(&bytes).expect(name).data,
            image.data,
            "{name}: no budget decodes the same pixels"
        );
    }
}

/// The budget travels in the source policy, so every resolver of a reference
/// — a data URI, a file inside the media directory, a CLI path — applies it.
#[test]
fn the_source_policy_carries_the_decode_budget_to_every_reference() {
    let png = hex(PNG_RGB8);
    let uri = format!("data:image/png;base64,{}", encode_base64_for_test(&png));

    let mut tight = ImageSourcePolicy::server(None, false);
    tight.max_source_pixels = Some(19);
    let err = load_image_source(&uri, &tight).expect_err("20 pixels, budget 19");
    assert_eq!(err.code(), "image_too_large", "{err}");
    tight.max_source_pixels = Some(20);
    assert_eq!(load_image_source(&uri, &tight).expect("fits").width, 5);

    // By default (and for a policy that sets none) only the hard limits.
    assert_eq!(ImageSourcePolicy::default().max_source_pixels, None);
    assert_eq!(ImageSourcePolicy::local_user().max_source_pixels, None);
    assert_eq!(
        ImageSourcePolicy::server(None, true).max_source_pixels,
        None
    );

    // `with_token_budget` derives it from the per-image token budget.
    let derived = ImageSourcePolicy::local_user().with_token_budget(1024);
    assert_eq!(derived.max_source_pixels, Some(16 * 1024 * 1024));
    assert!(derived.allow_any_local_path, "everything else is kept");
    let small = ImageSourcePolicy::server(None, false).with_token_budget(8);
    assert_eq!(
        small.max_source_pixels,
        Some(super::super::preprocess::MIN_SOURCE_PIXELS),
        "a small token budget never refuses ordinary photographs"
    );
    let largest = ImageSourcePolicy::server(None, false)
        .with_token_budget(super::super::preprocess::MAX_IMAGE_MAX_TOKENS);
    assert_eq!(largest.max_source_pixels, Some(MAX_DECODED_PIXELS));

    // A file inside the media directory is held to it as well.
    let dir = scratch_dir("decode-budget");
    std::fs::write(dir.join("pic.png"), &png).expect("fixture");
    let mut rooted = ImageSourcePolicy::server(Some(dir.clone()), false);
    rooted.max_source_pixels = Some(10);
    let err = load_image_source("file://pic.png", &rooted).expect_err("over the budget");
    assert_eq!(err.code(), "image_too_large", "{err}");
    rooted.max_source_pixels = None;
    assert_eq!(
        load_image_source("file://pic.png", &rooted)
            .expect("no budget")
            .width,
        5
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ── The bounded read of a local file ──────────────────────────────────────

/// A reader that yields more than its owner declared — a file that grew
/// after its length was read, or one whose length was never true (a pipe,
/// `/proc`): the read stops one byte past the limit instead of buffering it
/// all.
#[test]
fn a_read_never_buffers_more_than_the_limit_whatever_the_declared_length() {
    use std::io::Read as _;
    // Declares 10 bytes, would yield 1000: refused, and only `limit + 1`
    // bytes were ever taken from it.
    let mut lying = std::io::repeat(7).take(1000);
    let err = read_bounded(&mut lying, 10, 100).expect_err("outgrew the limit");
    assert_eq!(err, BoundedReadError::TooLarge(101));
    assert_eq!(
        lying.limit(),
        1000 - 101,
        "the read was capped at limit + 1"
    );

    // A declared length past the limit is refused without a read.
    let mut untouched = std::io::repeat(7).take(1000);
    assert_eq!(
        read_bounded(&mut untouched, 5_000, 100),
        Err(BoundedReadError::TooLarge(5_000))
    );
    assert_eq!(untouched.limit(), 1000, "nothing was read");

    // Exactly the limit is fine, one more is not; an honest short file reads whole.
    let exact = vec![3u8; 100];
    assert_eq!(read_bounded(&exact[..], 100, 100).expect("exact"), exact);
    let over = [3u8; 101];
    assert_eq!(
        read_bounded(&over[..], 100, 100),
        Err(BoundedReadError::TooLarge(101)),
        "a length that understates the file by one byte"
    );
    assert_eq!(read_bounded(&b"abc"[..], 3, 100).expect("short"), b"abc");
}

/// A file one byte past [`MAX_ENCODED_IMAGE_BYTES`] is refused from the open
/// handle's length alone (a sparse file: nothing is written or read), and a
/// path that is not a regular file is refused before it is opened.
#[test]
fn a_local_file_past_the_encoded_limit_or_not_a_file_is_refused() {
    let dir = scratch_dir("read-limited");
    let big = dir.join("big.png");
    let file = std::fs::File::create(&big).expect("create");
    file.set_len(MAX_ENCODED_IMAGE_BYTES as u64 + 1)
        .expect("a sparse file");
    drop(file);
    let policy = ImageSourcePolicy::local_user();
    let shown = big.to_string_lossy().into_owned();
    let err = load_image_bytes(&shown, &policy).expect_err("one byte over");
    assert_eq!(
        err,
        ImageInputError::EncodedTooLarge {
            bytes: MAX_ENCODED_IMAGE_BYTES + 1,
            limit: MAX_ENCODED_IMAGE_BYTES,
        }
    );
    assert_eq!(err.code(), "image_too_large");

    // Exactly at the limit is read (zeros are not an image: the decoder
    // says so, after the read succeeded).
    let at_limit = dir.join("at_limit.bin");
    let file = std::fs::File::create(&at_limit).expect("create");
    file.set_len(MAX_ENCODED_IMAGE_BYTES as u64)
        .expect("sparse");
    drop(file);
    let bytes = load_image_bytes(&at_limit.to_string_lossy(), &policy).expect("read whole");
    assert_eq!(bytes.len(), MAX_ENCODED_IMAGE_BYTES);

    // A directory is not a file.
    let shown = dir.to_string_lossy().into_owned();
    let err = load_image_bytes(&shown, &policy).expect_err("a directory");
    assert_eq!(err.code(), "image_file_unreadable", "{err}");
    assert!(err.to_string().contains("not a regular file"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
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

/// The refusal of an unrecognised format describes it without quoting it:
/// the bytes may be a non-image resource fetched for a client (a remote
/// `image_url`), and the message is returned to that client.
#[test]
fn an_unrecognised_format_is_refused_without_echoing_its_bytes() {
    let err = decode_image(b"TOP-SECRET-0123456789").expect_err("not an image");
    assert_eq!(err.code(), "image_format_unsupported");
    let message = err.to_string();
    assert!(
        message.contains("not a recognised image format"),
        "{message}"
    );
    for leaked in ["TOP", "SECRET", "54 4f", "544f", "0123"] {
        assert!(
            !message.contains(leaked),
            "{leaked:?} leaked into: {message}"
        );
    }
    // Whatever the unrecognised content is, the message is the same.
    let other = decode_image(b"{\"key\": \"another secret\"}").expect_err("not an image");
    assert_eq!(other.to_string(), message);
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
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
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

    // Remote URLs: refused without a fetcher, opted in or not, naming why.
    for policy in [&cli, &server, &ImageSourcePolicy::server(None, true)] {
        let err = load_image_source("https://example.com/cat.png", policy).expect_err("remote");
        assert_eq!(err.code(), "image_url_fetch_disabled");
    }
    // With a fetcher installed the reference goes to it, and its bytes are
    // decoded like any others.
    let fetching = ImageSourcePolicy::server(None, true).with_remote_fetcher(
        super::super::remote::SharedRemoteImageFetcher::new(std::sync::Arc::new(ServesBytes(hex(
            PNG_RGB8,
        )))),
    );
    assert_eq!(
        load_image_source("https://example.com/cat.png", &fetching)
            .expect("fetched")
            .width,
        5
    );
    let err = load_image_source("ftp://example.com/cat.png", &cli).expect_err("ftp");
    assert_eq!(err.code(), "image_url_scheme_unsupported");
    let err = load_image_source("   ", &cli).expect_err("blank");
    assert_eq!(err.code(), "image_url_scheme_unsupported");

    let _ = std::fs::remove_dir_all(&dir);
}

/// A unique scratch directory under the system temp dir.
fn scratch_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "oxibonsai-image-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

/// The media directory is a hard boundary: every way of naming a file
/// outside it — `..`, an absolute path, a bare `file://` — is a typed
/// `image_file_refused`, while `.` components and nesting stay inside.
#[test]
fn a_media_directory_refuses_every_spelling_of_a_path_outside_it() {
    let base = scratch_dir("media-escape");
    let root = base.join("media");
    std::fs::create_dir_all(root.join("a").join("b")).expect("media dir");
    std::fs::write(root.join("a").join("b").join("pic.png"), hex(PNG_RGB8)).expect("fixture");
    std::fs::write(base.join("secret.png"), hex(PNG_RGB8)).expect("secret");
    let outside = base.join("secret.png").to_string_lossy().into_owned();
    let policy = ImageSourcePolicy::server(Some(root.clone()), false);

    for inside in [
        "file://a/b/pic.png",
        "file://./a/./b/pic.png",
        "file://a/b/../b/pic.png",
    ] {
        // `..` is refused outright, even where the result would stay inside.
        let result = load_image_source(inside, &policy);
        if inside.contains("..") {
            let err = result.expect_err(inside);
            assert_eq!(err.code(), "image_file_refused", "{inside}: {err}");
        } else {
            assert_eq!(result.expect(inside).width, 5, "{inside}");
        }
    }
    for escape in [
        "file://../secret.png".to_string(),
        "file://a/../../secret.png".to_string(),
        "file://a/b/../../../secret.png".to_string(),
        format!("file://{outside}"),
        "file://".to_string(),
    ] {
        let err = load_image_source(&escape, &policy).expect_err(&escape);
        assert_eq!(err.code(), "image_file_refused", "{escape}: {err}");
    }
    let _ = std::fs::remove_dir_all(&base);
}

/// A symlink inside the media directory that leads out of it is refused
/// once resolved — whether it names the file itself or a directory on
/// the way — while a link that stays inside is followed.
#[cfg(unix)]
#[test]
fn a_symlink_out_of_the_media_directory_is_refused_with_a_typed_error() {
    use std::os::unix::fs::symlink;
    let base = scratch_dir("media-symlink");
    let root = base.join("media");
    let outside_dir = base.join("outside");
    std::fs::create_dir_all(&root).expect("media dir");
    std::fs::create_dir_all(&outside_dir).expect("outside dir");
    std::fs::write(outside_dir.join("secret.png"), hex(PNG_RGB8)).expect("secret");
    std::fs::write(root.join("real.png"), hex(PNG_RGB8)).expect("real");
    symlink(outside_dir.join("secret.png"), root.join("file_link.png")).expect("file link");
    symlink(&outside_dir, root.join("dir_link")).expect("dir link");
    symlink(root.join("real.png"), root.join("inside_link.png")).expect("inside link");
    let policy = ImageSourcePolicy::server(Some(root.clone()), false);

    for escape in ["file://file_link.png", "file://dir_link/secret.png"] {
        let err = load_image_source(escape, &policy).expect_err(escape);
        assert_eq!(err.code(), "image_file_refused", "{escape}: {err}");
        assert!(
            err.to_string()
                .contains("resolves outside the media directory"),
            "{escape}: {err}"
        );
    }
    assert_eq!(
        load_image_source("file://inside_link.png", &policy)
            .expect("a link that stays inside is followed")
            .width,
        5
    );
    // A media directory that is itself reached through a link is fine:
    // the boundary is the directory it resolves to.
    let via_link = base.join("media_link");
    symlink(&root, &via_link).expect("root link");
    let linked = ImageSourcePolicy::server(Some(via_link), false);
    assert_eq!(
        load_image_source("file://real.png", &linked)
            .expect("inside the resolved directory")
            .width,
        5
    );
    assert_eq!(
        load_image_source("file://dir_link/secret.png", &linked)
            .expect_err("still cannot leave it")
            .code(),
        "image_file_refused"
    );
    let _ = std::fs::remove_dir_all(&base);
}

/// A fetcher that answers every reference with the same bytes.
struct ServesBytes(Vec<u8>);

impl super::super::remote::RemoteImageFetcher for ServesBytes {
    fn fetch(&self, _url: &str, _max_bytes: usize) -> Result<Vec<u8>, ImageInputError> {
        Ok(self.0.clone())
    }
}

/// The refusals name the setting that fixes them: the flag and the
/// environment variable, once each — and say what the opt-in needs.
#[test]
fn the_refusal_texts_name_the_flag_and_the_environment_variable() {
    let remote = "https://example.com/cat.png";
    let default = load_image_bytes(remote, &ImageSourcePolicy::server(None, false))
        .expect_err("refused")
        .to_string();
    assert!(default.contains("--allow-image-url-fetch"), "{default}");
    assert!(default.contains("OXI_ALLOW_IMAGE_URL_FETCH=1"), "{default}");
    assert!(default.contains("server-side request forgery"), "{default}");
    assert!(
        default.contains("on a front end that installs a fetcher with an address policy"),
        "the opt-in alone is not promised to work: {default}"
    );
    assert!(default.contains("`oxibonsai` command"), "{default}");

    let opted_in = load_image_bytes(remote, &ImageSourcePolicy::server(None, true))
        .expect_err("refused: no fetcher installed")
        .to_string();
    assert!(
        opted_in.contains("remote fetching was opted in to"),
        "{opted_in}"
    );
    assert!(
        opted_in.contains("installed no fetcher with an address policy"),
        "{opted_in}"
    );
    assert!(opted_in.contains("with_remote_fetcher"), "{opted_in}");
    assert!(
        !opted_in.contains("OXI_ALLOW_IMAGE_URL_FETCH"),
        "an operator who opted in is not told to opt in: {opted_in}"
    );

    let local = load_image_bytes("images/cat.png", &ImageSourcePolicy::server(None, false))
        .expect_err("no media directory")
        .to_string();
    assert!(local.contains("--media-path <dir>"), "{local}");
    assert!(local.contains("OXI_MEDIA_PATH"), "{local}");
    assert!(local.contains("base64 data URI"), "{local}");
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
const PNG_GREY2_RGB: &str =
    "555555555555ffffffaaaaaa555555ffffff555555555555aaaaaa555555000000000000aaaaaa555555aaaaaa";
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
const PNG_ADAM7_PAL4_RGB: &str =
    "367e5c61b1d0218eee367e5c1b4dc7218eee856e2361b1d0856e2361b1d0218eee1b4dc70267286bdafb1b4dc7";
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
